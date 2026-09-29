import typing
from collections import defaultdict

import torch
from torch.optim import Optimizer
import numpy as np
import gymnasium as gym
from omegaconf import OmegaConf
from tensordict import TensorDict

from s2p.trainers.trainer_base import TrainerBase
from s2p.planners.planner_base import PlannerBase
from s2p.models.agent_model import AgentModel
from s2p.models.base.observation_policy_model_base import ObservationPolicyModelBase
from s2p.losses.loss_function_base import LossFunctionBase
from s2p.lib.transition_data import OnlineTransitionDataset
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.lib.utils import episode_boundary


class TeacherLatentCacheSpec(typing.NamedTuple):
	"""How one teacher encoder's output is cached into the replay buffer."""

	# Observation key the features are written to. Picking the key the encoder itself
	# short-circuits on is what lets the losses and the planner stay unaware of the cache.
	latent_key: str
	# Observation key the features are computed from. Dead weight once they are cached, so
	# it is dropped from the stored observation (see `drop_cached_teacher_inputs`).
	input_key: str
	# Dtype the features are stored as. They are cached, cast, and widened back on the way
	# out, so this trades buffer size against precision on the stored copy only.
	dtype: torch.dtype


# Teacher encoders whose output is a deterministic function of the observation, and is
# therefore worth computing once at collection time rather than on every replay of that
# frame. Keyed by class name so that adding an encoder here does not mean importing its
# module into the trainer; the lookup walks the MRO, so subclasses are covered too.
CACHEABLE_TEACHER_ENCODERS = {
	"DINOV3PassthroughEncoderModel": TeacherLatentCacheSpec(
		# The key `DINOV3PassthroughEncoderModel.encode` short-circuits on, and the one the
		# recorded ManiSkill datasets already carry.
		latent_key="dino_patch_features",
		input_key="rgb",
		# fp16 halves the stored features (2.71 MB rather than 5.42 MB per horizon-6 window,
		# so a 50k buffer costs 136 GB rather than 271 GB). Measured cost: ~1.8e-4 relative
		# error on the features, which moves the student latent by ~2e-5 of the sigma the
		# student's own rsample() already draws with, and the distillation loss by 6e-4 of
		# its batch-to-batch spread.
		dtype=torch.float16,
	),
}

def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)

class Every:
	def __init__(self, frequency:float):
		self.frequency = frequency
		self.count = 0

	def __call__(self):
		self.count += self.frequency
		updates = int(self.count)
		self.count -= updates
		return updates

class OnlineTrainer(TrainerBase):
	def __init__(self, 
			  cfg:OmegaConf, 
			  task:OnlineTaskBase, 
			  loss_functions:typing.Dict[str, LossFunctionBase], 
			  loss_function_weights:typing.Dict[str, float]
		):

		super().__init__(cfg, task, loss_functions, loss_function_weights)
		self.cfg = cfg

		# How many envs this trainer collects from, its own choice rather than the task's: each
		# online component builds its own env, so an evaluator can score on a different number.
		self.num_envs = int(self.cfg.num_envs)
		self.env = task.make_env(num_envs=self.num_envs)

		# `num_stacked_frames` must match the env's `FrameStack`, and is what lets the buffer
		# store `num_frames + horizon - 1` frames per window instead of `horizon * num_frames`.
		# The dataset verifies the overlap it implies on every append, so a value that
		# disagrees with the env raises on the first window rather than corrupting the buffer.
		self.dataset = OnlineTransitionDataset(
			capacity=self.cfg.buffer_size,
			horizon=self.cfg.horizon,
			load_device=self.cfg.load_device,
			batch_device=self.cfg.batch_device,
			num_stacked_frames=self.cfg.num_stacked_frames,
			storage_dir=self.cfg.buffer_storage_dir,
		)
		if self.cfg.seed_action_mode not in ("random", "agent"):
			raise ValueError("seed_action_mode must be one of ('random', 'agent').")

		self.last_obs = None
		self.previous_plan = None

		# Resolved from the model on the first observation we collect, since the trainer is
		# built before the model is handed to it. `None` means "not looked for yet";
		# an empty list means "looked, and this model has no cacheable teacher".
		self._teacher_latent_encoders = None

		# `update_to_data_ratio` is gradient steps per *env step*, so the collection frequency is
		# divided by `num_envs`: one call now returns `num_envs` windows, i.e. `num_envs * horizon`
		# env steps, and collecting that much every `num_envs` times as rarely leaves the ratio
		# where it was. That invariance is the point -- widening the env keeps the same training
		# recipe and buys back the wall clock, rather than quietly training on N times less data
		# per collected step.
		self.data_counter = Every(1/(self.cfg.update_to_data_ratio * self.num_envs))

	@property
	def training_dataset(self):
		return self.dataset
	
	@property
	def validation_dataset(self):
		return self.dataset

	def _resolve_teacher_latent_encoders(self, model:AgentModel):
		"""
		The `(encoder, spec)` pairs in `model` listed in `CACHEABLE_TEACHER_ENCODERS`.

		Only frozen encoders qualify: a trainable teacher's features go stale the moment its
		weights move, so a buffer full of them would be silently training the student
		against whatever the teacher happened to look like when each frame was collected.
		"""
		resolved = []
		for module in model.modules():
			spec = next(
				(CACHEABLE_TEACHER_ENCODERS[cls.__name__]
				 for cls in type(module).__mro__
				 if cls.__name__ in CACHEABLE_TEACHER_ENCODERS),
				None
			)
			if spec is None:
				continue

			if not module.cfg.freeze:
				raise ValueError(
					f"`cache_teacher_latents=True` requires a frozen teacher, but this "
					f"model's {type(module).__name__} has `freeze=False`. Caching a teacher "
					"that is still training would pin every stored frame to a stale encoding."
				)

			# The spec names the key to drop, but the encoder reads whichever key its own
			# config points at. If they disagree we would cache one key and delete another,
			# leaving the pixels in the buffer and the encoder without its input.
			observation_key = getattr(module.cfg, "observation_key", None)
			if observation_key is not None and observation_key != spec.input_key:
				raise ValueError(
					f"{type(module).__name__} reads observation key '{observation_key}', but "
					f"CACHEABLE_TEACHER_ENCODERS lists '{spec.input_key}' as its input key. "
					"Update the spec so the cached key and the dropped key agree."
				)

			resolved.append((module, spec))

		latent_keys = [spec.latent_key for _, spec in resolved]
		duplicates = {key for key in latent_keys if latent_keys.count(key) > 1}
		if duplicates:
			raise ValueError(
				f"Several cacheable teacher encoders in this model write the same "
				f"observation key(s) {sorted(duplicates)} and would overwrite each other. "
				"Give them distinct `latent_key`s in CACHEABLE_TEACHER_ENCODERS."
			)
		return resolved

	def _cache_teacher_latents(self, observation, model:AgentModel):
		"""
		Attach the frozen teacher's patch features to `observation`, in place, and drop the
		pixels they were computed from.

		Applied to every observation as it comes out of the env, so the features reach the
		replay buffer instead of the pixels. Every later encode of that observation - the
		planner's, on this very step, and every training batch the frame is replayed in -
		then short-circuits to them instead of re-running the ViT. The teacher is frozen,
		so this is exactly the same tensor it would have recomputed.

		Once the features are cached the pixels are dead weight: nothing downstream reads
		them, and they cost more to store than the features do (a horizon-6 window of
		frame-stacked 224x224 RGB is 2.71 MB against 5.42 MB of fp32 features). Dropping
		them is what keeps the buffer affordable. `drop_cached_teacher_inputs: False`
		keeps them, which is what an evaluator running over the training buffer and asking
		for pixel gradients (`force_pixel_gradients`, as SaliencyEvaluator does) needs.

		The features are left at full precision here rather than at `spec.dtype`, so that
		the planner's encode of this same observation - which short-circuits to them on
		this very step - hands the student the dtype its weights are in. They are narrowed
		only on the way into the buffer, in `_to_storage_dtype`.
		"""
		if not self.cfg.cache_teacher_latents:
			return observation

		if self._teacher_latent_encoders is None:
			self._teacher_latent_encoders = self._resolve_teacher_latent_encoders(model)

		for encoder, spec in self._teacher_latent_encoders:
			# `encode` builds a graph when the encoder is unfrozen; detach regardless so a
			# stored observation can never keep an autograd graph alive in the buffer.
			observation[spec.latent_key] = encoder.encode(observation).detach()

			# Deleting from the observation the env handed us is safe: `StackFrames` builds
			# this TensorDict fresh on every step and keeps its own deque of raw frames, so
			# the next stack is unaffected.
			if self.cfg.drop_cached_teacher_inputs:
				del observation[spec.input_key]
		return observation

	def _to_storage_dtype(self, subtrajectory_td:TensorDict):
		"""
		Narrow the cached latents to the dtype they are stored as, in place.

		Applied once per window on the way into the buffer, so the dtype the buffer
		allocates for the key (fixed by `_init_storage` from the first window it sees) is
		the storage dtype rather than the encoder's output dtype.
		"""
		for _, spec in self._teacher_latent_encoders or []:
			key = ("obs", spec.latent_key)
			if key in subtrajectory_td.keys(include_nested=True):
				subtrajectory_td[key] = subtrajectory_td[key].to(spec.dtype)
		return subtrajectory_td

	def _from_storage_dtype(self, batch:TensorDict):
		"""Widen the cached latents back to full precision on the way out of the buffer."""
		for _, spec in self._teacher_latent_encoders or []:
			key = ("obs", spec.latent_key)
			if key in batch.keys(include_nested=True):
				batch[key] = batch[key].float()
		return batch

	def _agent_action(self, obs, model:AgentModel):
		"""The agent's action for every env at once, chosen as `action_mode` says."""
		num_envs = self.num_envs
		with torch.no_grad():
			# `[T=1, num_envs, ...]`: the observation already carries the env axis,
			# so only the time axis has to be added.
			state = model.encoder_model.encode(obs.unsqueeze(0))
			if self.cfg.action_mode == "plan":
				if self.previous_plan is None:
					self.previous_plan = torch.zeros(size=(model.planner.cfg.horizon, num_envs, *self.task.action_dimension), device=self.cfg.device)
				action_prior = torch.zeros_like(self.previous_plan)
				action_prior[:-1] = self.previous_plan[1:]
				plan, _ = model.plan(state, action_prior)
				self.previous_plan = plan
				return plan[0]
			elif self.cfg.action_mode == "act":
				if isinstance(model.policy_model, ObservationPolicyModelBase):
					return model.act(obs)
				return model.act(state).squeeze(0)
			raise ValueError("action_mode must be one of ('plan', 'act').")

	def generate_subtrajectories(self, model:AgentModel):
		"""
		One horizon-length window per env, as a `[horizon, num_envs]` TensorDict.

		The envs step in lockstep, so the windows share a time axis; `add_subtrajectories` slices
		them back apart into one buffer entry each. A window never straddles a reset: the loop
		restarts it at an episode boundary, which under `episode_boundary` is a boundary for
		every env at once.
		"""

		# Parse the horizon which we want to generate the trajectory to
		horizon = int(self.cfg.horizon)
		num_envs = self.num_envs

		# Initialize the subtrajectory
		subtrajectory = {
			"obs": [],
			"action": [],
			"reward": [],
			"terminated": [],
			"truncated": [],
		}

		# Current observation is the last observation
		obs = self.last_obs
		# Not done yet
		done = False

		# Collect data until the subtrajectory is length `horizon`
		while len(subtrajectory["obs"]) < horizon:

			# If we hit a `done` state, or there's no observation, hit a reset
			# & initialize the subtrajectory dict (since we don't want a reset mid-trajectory)
			if done or obs is None:
				# Every env is reset: they step in lockstep and share one frame buffer.
				obs = self._cache_teacher_latents(self.env.reset().to(self.cfg.device), model)

				# The entry belonging to a reset observation has no action or reward that
				# produced it, so it is NaN for every env (the losses drop index 0 of both).
				subtrajectory = {
					"obs": [obs.detach().to(self.cfg.device)],
					"action": [torch.full_like(self.env.rand_act(), float('nan')).to(self.cfg.device)],
					"reward": [torch.full((num_envs, 1), float("nan"), dtype=torch.float32, device=self.cfg.device)],
					"terminated": [torch.zeros((num_envs, 1), dtype=torch.bool, device=self.cfg.device)],
					"truncated": [torch.zeros((num_envs, 1), dtype=torch.bool, device=self.cfg.device)],
				}
				self.previous_plan = None
				done = False

			# Else, step the env
			else:
				# Generate an action, for every env at once
				if len(self.dataset) < self.cfg.seed_steps and self.cfg.seed_action_mode == "random":
					action = self.env.rand_act()
				else:
					action = self._agent_action(obs, model)

				# Step the env
				obs, reward, terminated, truncated, _ = self.env.step(action.cpu())
				obs = self._cache_teacher_latents(obs.to(self.cfg.device), model)
				done = episode_boundary(terminated, truncated)

				# Add to the subtrajectory. Everything the env reports is already per-env, so
				# these only need the trailing feature axis the buffer stores them with.
				subtrajectory["obs"].append(obs.detach().to(self.cfg.device))
				subtrajectory["action"].append(action.detach().float().to(self.cfg.device))
				subtrajectory["reward"].append(reward.to(dtype=torch.float32, device=self.cfg.device).reshape(num_envs, 1))
				subtrajectory["terminated"].append(terminated.to(dtype=torch.bool, device=self.cfg.device).reshape(num_envs, 1))
				subtrajectory["truncated"].append(truncated.to(dtype=torch.bool, device=self.cfg.device).reshape(num_envs, 1))


		self.last_obs = obs

		subtrajectory_td = TensorDict(
			source={
				key: torch.stack(values, dim=0)
				for key, values in subtrajectory.items()
			},
			batch_size=[horizon, num_envs],
		)
		return subtrajectory_td


	def generate_data(self, model:AgentModel) -> None:
		for _ in range(self.data_counter()):
			with torch.no_grad():
				subtrajectory_td = self.generate_subtrajectories(model)
				self.dataset.add_subtrajectories(self._to_storage_dtype(subtrajectory_td))

	def fetch_batch(self):
		if len(self.dataset) < self.cfg.seed_steps:
			return None
		batch = torch.permute(
			self.dataset.sample(self.cfg.batch_size).to(self.cfg.device),
			dims=(1, 0)
		).contiguous()
		return self._from_storage_dtype(batch)

	def compute_step(self, batch, model:AgentModel, optimizer:Optimizer) -> typing.Dict[str, typing.Any]:
		optimizer.zero_grad()
		total_info = {}
		total_loss = torch.tensor(0.0, device=self.cfg.device)
		for loss_function_name, loss_function in self.loss_functions.items():
			loss, info = loss_function(batch=batch, model=model)
			total_loss += self.loss_function_weights[loss_function_name] * loss
			prefix = self._loss_prefixes[loss_function_name]
			total_info.update({prefix + key: value for key, value in info.items()})
		total_info["total_loss"] = total_loss.detach()
		total_loss.backward()
		optimizer.step()
		for name, sub_model in model.models.items():
			if sub_model is not None: sub_model.on_parameter_update_callback()
		return total_info
	
	
class DMControlWrapper(gym.Wrapper):
	def __init__(self, env, domain):
		self.env = env
		self.camera_id = 2 if domain == 'quadruped' else 0
		obs_shape = get_obs_shape(env)
		action_shape = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=np.full(action_shape, env.action_spec().minimum),
			high=np.full(action_shape, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.action_spec_dtype = env.action_spec().dtype

	@property
	def unwrapped(self):
		return self.env
	
	def _obs_to_array(self, obs):
		return torch.from_numpy(
			np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32))
	
	def reset(self):
		return self._obs_to_array(self.env.reset().observation)

	def step(self, action):
		reward = 0
		action = action.astype(self.action_spec_dtype)
		for _ in range(2):
			step = self.env.step(action)
			reward += step.reward
		return self._obs_to_array(step.observation), reward, False, False, defaultdict(float)
	
	def render(self, width=384, height=384, camera_id=None):
		return self.env.physics.render(height, width, camera_id or self.camera_id)

	def __getattr__(self, name):
		"""
		If this env does not have the attribute, then we try to 
		recursively access that attribute from inner envs.
		"""
		env = self.env
		while not hasattr(env, name):
			if hasattr(env, 'env'): # while the env is still wrapped,
				env = env.env
			else: # reached the innermost env and still didn't find it.
				raise AttributeError(f'{env} has no attribute {name}.')
		return getattr(env, name) # reached if env **has** attribute name.


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self):
		obs = self.env.reset()
		return self._obs_to_tensor(obs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		return self._obs_to_tensor(obs), reward, terminated, truncated, info

class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps
	
	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		self._t += 1
		truncated = truncated or self._t >= self.max_episode_steps
		return obs, reward, terminated, truncated, info

	def __getattr__(self, name):
		"""
		If this env does not have the attribute, then we try to 
		recursively access that attribute from inner envs.
		"""
		env = self.env
		while not hasattr(env, name):
			if hasattr(env, 'env'): # while the env is still wrapped,
				env = env.env
			else: # reached the innermost env and still didn't find it.
				raise AttributeError(f'{env} has no attribute {name}.')
		return getattr(env, name) # reached if env **has** attribute name.