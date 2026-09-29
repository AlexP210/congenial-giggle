from collections import defaultdict

from omegaconf import OmegaConf
import torch
import numpy as np
import gymnasium as gym

from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.tasks.base.single_env_batch import SingleEnvBatch
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.lib.transition_data import OfflineTransitionDataset

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
from tdmpc2.envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from tdmpc2.common import TASK_SET

def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)

class DMControlTask(OnlineTaskBase, OfflineTaskBase):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.cfg = cfg

		self.env = self.make_env()
		# Per-env, i.e. with the batch axis `SingleEnvBatch` adds dropped; see
		# `OnlineTaskBase._per_env_dims_from_space`.
		self._action_dimension = self._per_env_dims_from_space(self.env.action_space)
		# dm_control neither clips nor validates -- an out-of-spec control reaches MuJoCo
		# verbatim -- so respecting the `action_spec` bounds is left to whoever acts.
		self._action_limits = self._per_env_limits_from_space(self.env.action_space)
		self._observation_dimension = self._per_env_dims_from_space(self.env.observation_space)
		self._episode_length = 500
		self._task_name = self.cfg.task_name

	def make_env(self, num_envs: int = None, max_episode_steps: int = None):
		"""One env, presented as a batch of one; see `OnlineTaskBase.make_env`.

		dm_control steps a single simulator, so there is nothing to run in parallel here:
		anything above one env is refused rather than silently served as one. `SingleEnvBatch`
		sits outermost, so every adapter underneath keeps serving unbatched observations and
		only the outside world sees the batch axis.
		"""
		if num_envs is not None and int(num_envs) != 1:
			raise ValueError(
				f"DMControlTask runs a single dm_control simulator and cannot step "
				f"{num_envs} envs in parallel. Use a ManiSkill task for parallel collection, or "
				"ask this one for num_envs=1."
			)
		domain, task = self.cfg.task_name.replace('-', '_').split('_', 1)
		if (domain, task) not in suite.ALL_TASKS:
			raise ValueError('Unknown task:', task)
		env = suite.load(domain,
						task,
						task_kwargs={'random': self.cfg.seed},
						visualize_reward=False)
		env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
		env = DMControlWrapper(env, domain)
		# dm_control has no action chunking, so the limit is in the same steps the loop counts.
		env = Timeout(env, max_episode_steps=500 if max_episode_steps is None else int(max_episode_steps))
		env = TensorWrapper(env)
		return SingleEnvBatch(env)
	
	def get_control_interval(self):
		return 0.02
	
	def make_dataset(self):

		_custom_task_mapping = {
		}
		
		task_name_for_data_generation = self.cfg.task_name
		if self.cfg.task_name in _custom_task_mapping:
			task_name_for_data_generation  = _custom_task_mapping[self.cfg.task_name]

		training_set_exists = os.path.exists(self.cfg.training_data_path)
		validation_set_exists = os.path.exists(self.cfg.validation_data_path)
		
		if not (training_set_exists and validation_set_exists):
			training_data_root = os.path.dirname(self.cfg.training_data_path)
			validation_data_root = os.path.dirname(self.cfg.validation_data_path)
			full_dataset = OfflineTransitionDataset(
				paths=[os.path.join(training_data_root, f"chunk_{i}.pt") for i in range(4)], fraction=1.0
			)

			task_index = TASK_SET["mt30"].index(task_name_for_data_generation)
			
			action_ndim = self.action_dimension[-1]
			obs_ndim = self.observation_dimension[-1]

			tensordict = full_dataset.data
			mask_indices = tensordict["task"][:, 0, 0] == task_index
			filtered = tensordict[ mask_indices ]
			filtered["action"] = filtered["action"][...,:action_ndim]
			filtered["obs"] = filtered["obs"][...,:obs_ndim]
			torch.save(filtered, os.path.join(training_data_root, f"mt30-{self.cfg.task_name}.pt"))
			dataset = OfflineTransitionDataset(
				paths=[os.path.join(training_data_root, f"mt30-{self.cfg.task_name}.pt"),], fraction=1.0
			)
			tensordict = dataset.data
			validation_mask = torch.rand(tensordict.shape[0]) < 0.2
			validation = tensordict[validation_mask]
			train = tensordict[~validation_mask]
			torch.save(validation, os.path.join(validation_data_root, f"mt30-{self.cfg.task_name}-validation.pt"))
			torch.save(train, os.path.join(training_data_root, f"mt30-{self.cfg.task_name}-training.pt"))

		training_dataset =  OfflineTransitionDataset(
			paths=[self.cfg.training_data_path,], 
			horizon=self.cfg.horizon,
			fraction=self.cfg.fraction,
			overlap_ratio=self.cfg.overlap_ratio,
			load_device=self.cfg.load_device,
			batch_device=self.cfg.batch_device
		)
		
		validation_dataset =  OfflineTransitionDataset(
			paths=[self.cfg.validation_data_path,], 
			horizon=self.cfg.horizon,
			fraction=self.cfg.fraction,
			overlap_ratio=self.cfg.overlap_ratio,
			load_device=self.cfg.load_device,
			batch_device=self.cfg.batch_device
		)

		return training_dataset, validation_dataset

	
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

if __name__ == "__main__":
	task = DMControlTask(
		cfg=OmegaConf.create(
			{
				"task_name": "cheetah-jump",
				"data_root": "/path/to/tdmpc2_data/datasets/mt30/",
				"seed": 1,
				"horizon": 4,
				"fraction": 1.0,
				"overlap_ratio": 0.0,
				"load_device": "cpu",
				"batch_device": "cpu"
			}
      )
    )
	env = task.make_env()
	print(env.control_timestep())

