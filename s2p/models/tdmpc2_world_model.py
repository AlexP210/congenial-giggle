import typing
import copy
import hashlib

import torch
from omegaconf import OmegaConf
from tdmpc2.agent import TDMPC2
from tdmpc2.common import math
from tdmpc2.common.parser import parse_cfg
from tdmpc2.envs import make_env
from tdmpc2.envs.custom_maniskill import CUSTOM_MANISKILL_TASKS, PROPRIO_KEYS, VISUAL_OBS
from tdmpc2.common import TASK_SET

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.policy_model_base import PolicyModelBase

from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference

class TDMPC2WorldModel(
	EncoderModelBase, 
	DynamicsModelBase, 
	RewardModelBase, 
	StateValueModelBase, 
	PolicyModelBase,
	torch.nn.Module
):

	def __init__(self, cfg, task:TaskBase):
		super().__init__(cfg=cfg)
		self.cfg = cfg
		self.task = task
		
		# Mapping so we can get task embedding for custom tasks
		_custom_task_mapping = {
			"cheetah-flip": "cheetah-jump",
		}

		# Set the task index; should be 0 except for multi-task WMs
		self.task_index = None
		if self.cfg.tdmpc2_cfg.multitask:
			# Task is an original mt30 task
			if self.task.cfg.task_name in TASK_SET["mt30"]:
				self.task_index = TASK_SET["mt30"].index(self.task.cfg.task_name)
			# Task is not an original mt30 task, but has an equivalent embedding defined
			elif self.task.cfg.task_name in _custom_task_mapping:
				self.task_index = TASK_SET["mt30"].index(_custom_task_mapping[self.task.cfg.task_name])
			# Task is not an original mt30 task, nor has an equivalent embedding defined
			else:
				raise ValueError(
					f"Task {self.task.cfg.task_name} does not have an embedding ID, neither in original TDMPC2"
					 "MT30 dataset nor is it registered as a custom task")
		else:
			self.task_index = None

		OmegaConf.resolve(cfg.tdmpc2_cfg)
		self.parsed_tdmpc2_cfg = parse_cfg(cfg.tdmpc2_cfg)

		# HACK to make this work for both the original DMC state-based work
		# and the updated ManiSkill vision-based work.
		#
		# A vision-based checkpoint's obs_shape is read off the S2P task rather than off a TD-MPC2
		# env built here: that env would be a second sapien scene in this process, on the
		# physx_cuda backend TD-MPC2 hardcodes, and GPU PhysX can only be enabled once per process
		# -- so it breaks any task that has already built a physx_cpu one.
		if self.parsed_tdmpc2_cfg.obs in VISUAL_OBS:
			self._set_visual_obs_shape(task)
			self.parsed_tdmpc2_cfg.action_dim = task._action_dimension[-1]
			self.parsed_tdmpc2_cfg.action_dims = task._action_dimension[-1]
			self.parsed_tdmpc2_cfg.episode_length = task.episode_length
			self.parsed_tdmpc2_cfg.seed_steps = max(1000, 5*self.parsed_tdmpc2_cfg.episode_length)
		else:
			self._set_state_obs_shape(task)


		self.tdmpc2 = TDMPC2(self.parsed_tdmpc2_cfg)

		if self.parsed_tdmpc2_cfg.checkpoint is not None:
			self.load_from_file(self.parsed_tdmpc2_cfg.checkpoint)

	def _set_state_obs_shape(self, task):
		try:
			_ = make_env(self.parsed_tdmpc2_cfg) # Call this for the side-effect of populating the cfg
		except:
			self.parsed_tdmpc2_cfg.obs_shape = {"state": (1, )}
			self.parsed_tdmpc2_cfg.action_dim = task._action_dimension[-1]
			self.parsed_tdmpc2_cfg.action_dims = task._action_dimension[-1]
			self.parsed_tdmpc2_cfg.episode_length = task.episode_length
			self.parsed_tdmpc2_cfg.seed_steps = max(1000, 5*self.parsed_tdmpc2_cfg.episode_length)

	def _set_visual_obs_shape(self, task):
		"""`obs_shape` for an `rgb` / `rgb_state` checkpoint, in the layout TD-MPC2 trained on.

		TD-MPC2's `Pixels` wrapper concatenates frames into the channel axis, so the image stream
		is `(num_frames*C, H, W)` of S2P's per-env `(num_frames, C, H, W)`. `layers.conv` picks
		its conv stack from H, so the task has to render at the checkpoint's `camera_resolution`
		for the state_dict to load at all.

		Under `rgb_state` the proprio stream is `PROPRIO_KEYS[tdmpc2_cfg.task]` -- ManiSkill paths
		such as `agent/qpos` -- concatenated in that order. Each is found in the task's own
		observation by its `dataset_structure` path (`obs/agent/qpos`), so a task that does not
		expose one raises here rather than feeding the fuse layer a vector in the wrong order.
		"""
		tdmpc2_task = self.parsed_tdmpc2_cfg.task
		if tdmpc2_task in CUSTOM_MANISKILL_TASKS:
			trained_env = CUSTOM_MANISKILL_TASKS[tdmpc2_task]['env']
			if trained_env != task.cfg.task_name:
				raise ValueError(
					f'tdmpc2_cfg.task={tdmpc2_task!r} was trained on {trained_env}, but the S2P '
					f'task is {task.cfg.task_name}.')

		obs_dims = task._observation_dimension
		frames, channels, height, width = obs_dims[self.cfg.observation_key]
		obs_shape = {'rgb': (frames*channels, height, width)}

		self._proprio_leaves = None
		if self.parsed_tdmpc2_cfg.obs == 'rgb_state':
			if tdmpc2_task not in PROPRIO_KEYS:
				raise ValueError(
					f"obs='rgb_state' reads its proprio fields from PROPRIO_KEYS, which has no entry "
					f'for tdmpc2_cfg.task={tdmpc2_task!r}; expected one of {sorted(PROPRIO_KEYS)}.')
			nickname_of = {
				path: nickname
				for nickname, path in OmegaConf.to_container(task.cfg.dataset_structure)['obs'].items()
			}
			missing = [key for key in PROPRIO_KEYS[tdmpc2_task] if f'obs/{key}' not in nickname_of]
			if missing:
				raise ValueError(
					f'The TD-MPC2 {tdmpc2_task} checkpoint reads proprio fields {missing}, which '
					f'task.cfg.dataset_structure.obs does not expose (as obs/<field>).')
			self._proprio_leaves = tuple(nickname_of[f'obs/{key}'] for key in PROPRIO_KEYS[tdmpc2_task])
			obs_shape['state'] = (sum(obs_dims[leaf][-1] for leaf in self._proprio_leaves),)

		self.parsed_tdmpc2_cfg.obs_shape = obs_shape

	def _pad_observations(self, observation):
		# If this is a multi-task model, we need to pad the observations to the model's native size
		if self.parsed_tdmpc2_cfg.multitask:
			obs_type = self.parsed_tdmpc2_cfg.obs
			obs_dims = self.parsed_tdmpc2_cfg.obs_shape[obs_type]
			obs_ndim = len(obs_dims)
			batch_dims = observation.shape[:-obs_ndim]
			
			target_shape = (*batch_dims, *obs_dims)
			padded_observation = torch.zeros(size=target_shape, dtype=observation.dtype, device=observation.device)
			slices = tuple(slice(0, s) for s in observation.shape)
			padded_observation[slices] = observation
			observation = padded_observation
		return observation
	
	def _pad_actions(self, action):
		T, B = action.shape[:2]
		A = action.shape[2:]

		# If this is a multi-task model, we need to pad the observations to the model's native size
		if self.parsed_tdmpc2_cfg.multitask:
			action_dims = (self.parsed_tdmpc2_cfg.action_dim,) # hard coded to accept actions with ndim=1
			action_ndim = len(action_dims)
			batch_dims = action.shape[:-action_ndim]

			target_shape = (*batch_dims, *action_dims)
			padded_action = torch.zeros(size=target_shape, dtype=action.dtype, device=action.device)
			slices = tuple(slice(0, s) for s in action.shape)
			padded_action[slices] = action
			action = padded_action
		return action

	
	def _to_tdmpc2_observation(self, observation):
		"""S2P's observation TensorDict as the stream(s) TD-MPC2's encoder was trained on.

		- The image: S2P's `FrameStack` puts frames on their own axis, `(..., S, C, H, W)`, while
		  TD-MPC2's `Pixels` concatenated them into the channel axis, `(..., S*C, H, W)`. Both
		  stack oldest-first out of a `deque` and both hold `S` copies of the first frame after a
		  reset, so folding the two axes together is a plain reshape and not a permute. The cast
		  only changes dtype: `PixelPreprocess` divides by 255 itself.
		- The proprio vector (`rgb_state` only): the newest frame of each proprio leaf, since
		  `Pixels` stacked the camera alone and passed the current step's proprio through.
		"""
		rgb = observation[self.cfg.observation_key]
		*batch_dims, frames, channels, height, width = rgb.shape
		folded_dims = (frames*channels, height, width)
		trained_dims = tuple(self.parsed_tdmpc2_cfg.obs_shape['rgb'])
		if folded_dims != trained_dims:
			raise ValueError(
				f'Cannot encode a {tuple(rgb.shape)} observation with a TD-MPC2 encoder trained on '
				f'{trained_dims}: {frames} frames of {channels}x{height}x{width} fold to '
				f'{folded_dims}. Check the task\'s num_frames and visual_observation_resolution '
				f'against the checkpoint.')
		rgb = rgb.reshape(*batch_dims, *folded_dims).float()
		if self._proprio_leaves is None:
			return rgb, tuple(batch_dims)
		state = torch.cat(
			[observation[leaf][..., -1, :].float() for leaf in self._proprio_leaves], dim=-1)
		return {'rgb': rgb, 'state': state}, tuple(batch_dims)

	def _encode_visual(self, observation):
		observation, batch_dims = self._to_tdmpc2_observation(observation)
		# TD-MPC2's conv encoder takes a single leading batch dim, so flatten any extra batch dims
		# (e.g. time) down to one before encoding, then re-expand.
		if isinstance(observation, dict):
			flat = {
				key: value.reshape(-1, *self.parsed_tdmpc2_cfg.obs_shape[key])
				for key, value in observation.items()
			}
		else:
			flat = observation.reshape(-1, *self.parsed_tdmpc2_cfg.obs_shape['rgb'])
		encoded = self.tdmpc2.model.encode(flat, self.task_index)
		return encoded.reshape(*batch_dims, encoded.shape[-1])

	def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None):
		if self.parsed_tdmpc2_cfg.obs in VISUAL_OBS:
			return self._encode_visual(observation)
		observation = self._pad_observations(observation)

		# # Create the task index; TD-MPC2 expects one task index per batch element
		# # it handles the repeat over time inside the WM, if observation has ndim=3

		# Call the actual TDMPC2 encoder
		obs_type = self.parsed_tdmpc2_cfg.obs
		obs_dims = self.parsed_tdmpc2_cfg.obs_shape[obs_type]
		obs_ndim = len(obs_dims)
		batch_dims = observation.shape[:-obs_ndim]
		if not batch_dims: # We passed a single observation - al
			observation = observation.unsqueeze(0)
			batch_dims = observation.shape[:-obs_ndim]

		# TD-MPC2's encoder only supports a single leading batch dim, so flatten any
		# extra batch dims (e.g. time) down to one before encoding, then re-expand.
		flat_observation = observation.reshape(-1, *obs_dims)
		encoded = self.tdmpc2.model.encode(flat_observation, self.task_index)
		latent_dim = encoded.shape[-1]
		return encoded.reshape(*batch_dims, latent_dim)
	
	def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		obs_shape = self.parsed_tdmpc2_cfg.obs_shape
		if self.parsed_tdmpc2_cfg.obs == 'rgb':
			# the conv net never sees the task embedding -- that is only concatenated onto the
			# latent later, for dynamics/reward/pi/Qs
			return tuple(obs_shape['rgb']), self.tdmpc2.model._encoder['rgb']
		if self.parsed_tdmpc2_cfg.obs == 'rgb_state':
			module = _FusedEncodingFunction(self.tdmpc2.model, obs_shape['rgb'], obs_shape['state'])
			return (module.input_dim,), module
		return (
			[self.parsed_tdmpc2_cfg.obs_shape[self.parsed_tdmpc2_cfg.obs][0]+self.parsed_tdmpc2_cfg.task_dim,],
			self.tdmpc2.model._encoder[self.parsed_tdmpc2_cfg.obs]
		)
	
	def dynamics(self, s, a):
		a = self._pad_actions(a)
		action_dims = (self.parsed_tdmpc2_cfg.action_dim,) # hard coded to accept actions with ndim=1
		action_ndim = len(action_dims)
		batch_dims = a.shape[:-action_ndim]
		last_batch_dim = batch_dims[-1]

		if self.parsed_tdmpc2_cfg.multitask:
			task = torch.full(size=(last_batch_dim,), fill_value=self.task_index, device=a.device)
		else:
			task = None
		# Create the task index; TD-MPC2 expects one task index per batch element 
		# it handles the repeat over time inside the WM, if observation has ndim=3
		# task = torch.full(size=(B,), fill_value=self.task_index, device=s.device)
		return self.tdmpc2.model.next(s, a, task)
	
	def get_dynamics_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		return (
			(self.parsed_tdmpc2_cfg.latent_dim + self.parsed_tdmpc2_cfg.action_dim + self.parsed_tdmpc2_cfg.task_dim,),
			self.tdmpc2.model._dynamics
		)

	def reward(self, s, a):
		a = self._pad_actions(a)
		# T, B = s.shape[:2]
		action_dims = (self.parsed_tdmpc2_cfg.action_dim,) # hard coded to accept actions with ndim=1
		action_ndim = len(action_dims)
		batch_dims = a.shape[:-action_ndim]
		last_batch_dim = batch_dims[-1]

		if self.parsed_tdmpc2_cfg.multitask:
			task = torch.full(size=(last_batch_dim,), fill_value=self.task_index, device=a.device)
		else:
			task = None

		return math.two_hot_inv(self.tdmpc2.model.reward(s, a, task), self.parsed_tdmpc2_cfg)
	
	def get_reward_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		return (
			(self.parsed_tdmpc2_cfg.latent_dim + self.parsed_tdmpc2_cfg.action_dim + self.parsed_tdmpc2_cfg.task_dim,),
			self.tdmpc2.model._reward
		)
	
	def state_value(self, s):
		# T, B = s.shape[:2]
		latent_dims = (self.parsed_tdmpc2_cfg.latent_dim,) # hard coded to accept actions with ndim=1
		latent_ndim = len(latent_dims)
		batch_dims = s.shape[:-latent_ndim]
		last_batch_dim = batch_dims[-1]

		if self.parsed_tdmpc2_cfg.multitask:
			task = torch.full(size=(last_batch_dim,), fill_value=self.task_index, device=s.device)
		else:
			task = None
		action, _ = self.tdmpc2.model.pi(s, task)
		return self.tdmpc2.model.Q(s, action, task, return_type='avg')
	
	def get_state_value_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		params_td = self.tdmpc2.model._Qs.params[0]
		module_template = self.tdmpc2.model._Qs.module

		# 1. Flatten the TensorDict: {"0": {"weight": T}} -> {"0.weight": T}
		flat_params = params_td.flatten_keys(".")

		# 2. Forcefully attach them to the empty module
		hydrate_module(module_template, flat_params)

		return (
			(self.parsed_tdmpc2_cfg.latent_dim + self.parsed_tdmpc2_cfg.action_dim + self.parsed_tdmpc2_cfg.task_dim,),
			module_template
		)
	
	def policy(self, s):
		latent_dims = (self.parsed_tdmpc2_cfg.latent_dim,) # hard coded to accept actions with ndim=1
		latent_ndim = len(latent_dims)
		batch_dims = s.shape[:-latent_ndim]
		last_batch_dim = batch_dims[-1]

		if self.parsed_tdmpc2_cfg.multitask:
			task = torch.full(size=(last_batch_dim,), fill_value=self.task_index, device=s.device)
		else:
			task = None
		action, _ = self.tdmpc2.model.pi(s, task)
		return action
	
	def get_policy_function(self):
		return (
			(self.parsed_tdmpc2_cfg.latent_dim + self.parsed_tdmpc2_cfg.task_dim,),
			self.tdmpc2.model._pi
		)
	
	def save_to_file(self, filepath) -> None:
		# Frozen: these weights are still exactly the checkpoint this was built from, so
		# point at it instead of copying it. See s2p.lib.checkpointing.
		if self.cfg.freeze and self.parsed_tdmpc2_cfg.checkpoint is not None:
			save_checkpoint_reference(filepath, self.parsed_tdmpc2_cfg.checkpoint, type(self).__name__)
			return

		self.tdmpc2.save(filepath)

	def load_from_file(self, filepath:str) -> None:
		# May be a reference written by `save_to_file` above rather than weights.
		filepath = resolve_checkpoint_path(filepath)
		self.tdmpc2.load(filepath)

	def requires_grad_(self, requires_grad):
		return super().requires_grad_(requires_grad and not self.cfg.freeze)
	
class _FusedEncodingFunction(torch.nn.Module):
	"""An `rgb_state` encoder as the one-tensor module `get_encoding_function` promises.

	The FLOPs evaluator calls an encoding function on a single `(1, *input_dims)` tensor, but an
	`rgb_state` encoder is two streams and a fuse layer. So the input is the flattened image
	followed by the proprio vector, split back apart here, and the count covers all three parts.

	The world model is held *outside* the module tree (`object.__setattr__`) and only the
	streams and fuse layer `encode` runs are registered: calflops walks every submodule, and the
	Q ensemble's `TensorDictParams` break that walk -- besides which the dynamics, reward, pi and
	Qs would otherwise be counted into the encoder's parameters. Same reasoning as
	`_DINOWMRewardHeadForFLOPS`.
	"""

	def __init__(self, world_model, rgb_shape, state_shape):
		super().__init__()
		object.__setattr__(self, "world_model", world_model)
		self.encoder = world_model._encoder
		self.fuse = world_model._fuse
		self.rgb_shape = tuple(rgb_shape)
		self.rgb_dim = int(torch.Size(self.rgb_shape).numel())
		self.input_dim = self.rgb_dim + int(state_shape[0])

	def forward(self, x):
		return self.world_model.encode({
			'rgb': x[:, :self.rgb_dim].reshape(-1, *self.rgb_shape),
			'state': x[:, self.rgb_dim:],
		}, None)


def hydrate_module(module, flat_params):
	for name, tensor in flat_params.items():
		sub = module
		parts = name.split(".")
		
		# Navigate to the specific submodule (e.g., "0" -> "ln")
		for part in parts[:-1]:
			if part.isdigit():
				sub = sub[int(part)]
			else:
				sub = getattr(sub, part)
		
		# Determine the leaf name (e.g., "weight")
		leaf_name = parts[-1]
		
		# Permanently attach the tensor as a Parameter
		# We detach and clone to ensure it's a clean copy for evaluation
		param = torch.nn.Parameter(tensor.detach().clone())
		setattr(sub, leaf_name, param)


_registry: dict = {}

def get_or_create(cfg, task) -> "TDMPC2WorldModel":
	key = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = TDMPC2WorldModel(cfg=cfg, task=task)
	return _registry[key]