import abc
import typing
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import deque
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.planners.planner_base import PlannerBase
from s2p.models.plannable_model_base import PlannableModelBase

# from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False})
# import custom_isaaclab_tasks.tasks  # noqa: F401
# from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class IsaacLabEvaluator(OnlineEvaluatorBase):
	def __init__(self, cfg, planner:PlannerBase):

		super().__init__(cfg)
		self.cfg = cfg

		self.planner = planner
		self.env = None
		self._start_sim()

	def _start_sim(self):
		# Load the IsaacLab AppLauncher
		# NOTE: This needs to happen before importing `custom_isaaclab_tasks`, since otherwise you get
		# errors for `omni module not found`.
		from isaaclab.app import AppLauncher
		app_launcher = AppLauncher(launcher_args={
			"livestream": int(self.cfg.visualize),
			"enable_cameras": self.cfg.enable_cameras,
			"device": self.cfg.device,
			"headless": not self.cfg.visualize
		})
		self.simulation_app = app_launcher.app
		del AppLauncher

		# Import our custom tasks to register the tasks to `gym`
		# NOTE: This has to happen here, because it needs to go after the simulation app
		# is created; and the simulation app has to get created here since we want to control its
		# creation from `evaluator_cfg`. 
		import custom_isaaclab_tasks.tasks
		self.tasks_module = custom_isaaclab_tasks.tasks

		# Grab the environment CFG file
		from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
		# Update the env_cfg based on the `evaluator_cfg`
		self.env_cfg = load_cfg_from_registry(self.cfg.env_name, "env_cfg_entry_point")
		self.env_cfg.scene.num_envs = self.cfg.num_envs
		self.env_cfg.device = self.cfg.device
		self.env_cfg.sim.device = self.cfg.device
		self.env_cfg.task_index = self.cfg.task_index
		self.env_cfg.seed = self.cfg.seed

		# Create the env
		self.env = gym.make(self.cfg.env_name, cfg=self.env_cfg, render_mode="rgb_array")
		self.steps_per_episode = int((self.env_cfg.episode_length_s / self.env_cfg.sim.dt) // self.env_cfg.decimation)
		  
		# The env was initially developed to work well with TD-MPC2 for training our first Teachers;
		# TD-MPC2 has some wrapper classes around the env. So we'll add them here to not break compatibility
		# with the teachers we've already trained.
		self.env = IsaacLabWrapper(env=self.env, cfg=self.cfg, task_name=self.cfg.env_name)
		if self.cfg.observation_type == "rgb":
			self.env = Pixels(env=self.env, cfg=self.cfg)
		self.env = TensorWrapper(self.env)

	def __call__(self, model:PlannableModelBase) -> typing.Dict[str, typing.Any]:
		info = {}

		# Reset the environment
		(parallel_obs, parallel_info), parallel_done, parallel_ep_reward, t = (
			self.env.reset(),
			torch.full(size=(self.cfg.num_envs,1),fill_value=False, device=self.cfg.device),
			torch.full(size=(self.cfg.num_envs,1),fill_value=0.0, device=self.cfg.device),
			0,
		)
		# TODO
		# if self.cfg.save_video:
		# 	self.logger.video.init(self.env, enabled=(i == 0))
		previous_plans = torch.zeros(size=(self.cfg.num_envs, self.planner.cfg.horizon, self.planner.cfg.action_dim), device=self.cfg.device)
		frames = []
		for _ in tqdm(range(self.steps_per_episode), desc="IsaacLab Evaluator"):
			if self.cfg.save_video: frames.append(self.env.render())
			# Find which ones are not done yet
			not_done_envs = torch.nonzero(torch.logical_not(parallel_done), as_tuple=True)[0].tolist()

			# Compute the actions for those that are not done yet
			parallel_actions = torch.zeros(
				size=(self.cfg.num_envs, self.planner.cfg.action_dim),
				device=self.cfg.device
			)
			
			for env_id in not_done_envs:

				previous_plan = previous_plans[env_id]
				action_prior = torch.zeros_like(previous_plan)
				action_prior[:-1] = previous_plan[1:]
				obs = parallel_obs[env_id].to(self.cfg.device)
				
				current_state = model.encode(obs.unsqueeze(0))
				plan, _ = self.planner.plan(model, current_state, True, action_prior)
				previous_plans[env_id] = plan
				action = plan[0]
				parallel_actions[env_id] = action
			# Apply the actions for all of them (only non-zero for those that are not done)
			(
				parallel_obs,
				parallel_reward,
				parallel_terminated,
				parallel_truncated,
				env_info,
			) = self.env.step(parallel_actions.cpu())

			# For each env that was just updated, add the reward
			for env_id in not_done_envs:
				parallel_ep_reward[env_id] += parallel_reward[env_id]

			# # Record a frame of video
			# if self.cfg.save_video:
			# 	self.logger.video.record(self.env)

			# Find which envs are done
			parallel_done = parallel_terminated | parallel_truncated

			if parallel_done.all(): break
		frames = np.stack(frames)
		frames = frames.transpose(0, 3, 1, 2)
		return_ = parallel_ep_reward.cpu().numpy().squeeze()
		success = env_info["successes"].float().cpu().numpy().squeeze()
		info.update({
			f"episode_return": return_.mean(),
			f"episode_return_sem": return_.std()/np.sqrt(len(return_)),
			f"episode_return_distribution": return_,
			f"episode_success_rate": success.mean(),
			f"episode_success_rate_sem": success.mean() * (1 - success.mean()) / np.sqrt(len(success)),
			f"episode_success_rate_distribution": success,
		})
		if self.cfg.save_video:
			info.update({
				f"video": frames
			})
		# self._stop_sim()
		return info
	
	def _stop_sim(self):
		self.env.close()
		self.simulation_app.close()


class IsaacLabWrapper(gym.Wrapper):
	def __init__(self, env, cfg, task_name):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.task_name = task_name

	def reset(self, env_id=None):
		if env_id is None:
			obs, _ = self.env.reset()
		else:
			self.env.unwrapped._reset_idx(
				env_ids=torch.tensor(
					[
						env_id,
					]
				)
			)
			obs = self.env.unwrapped._get_observations()
		if type(obs) == type(dict()):
			obs = obs[self.cfg.observation_type]
		return self.obs_to_cpu(obs)

	def step(self, action):
		action = torch.from_numpy(action)
		obs, reward, terminated, truncated, info = self.env.step(action)
		 # If the env is returning us a dict
		if type(obs) == type(dict()):
			obs = obs[self.cfg.observation_type]

		if "Manager" in self.task_name:
			# For the manager-based envs, we need to get the termination and truncation
			# signal from the `info` dict; if we do it normally, then the ManagerBasedRLEnv
			# will automatically reset specific envs that have the termination or truncated flags
			# For consistency with the FactoryEnv's, we want to reset them all together instead so 
			# can't do this
			terminated = torch.full_like(terminated, fill_value=False)
			truncated = torch.full_like(truncated, fill_value=False)
			for key, val in info.items():
				if key.split("/")[0] == "termination": terminated |= val
				elif key.split("/")[0] == "truncation": truncated |= val
			# Shuffle around the info dict
			successes = info["successes"]
			episode_length = info["episode_lengths"]
			info = {}
			info["successes"] = successes
			info["episode_lengths"] = episode_length
		return_value = (
			self.obs_to_cpu(obs),
			reward.cpu(),
			terminated.cpu(),
			truncated.cpu(),
			self.info_to_cpu(info),
		)
		return return_value

	def render(self):
		return self.env.render()

	def info_to_cpu(self, info):
		return {key: val.cpu()  for key, val in info.items()}

	def obs_to_cpu(self, obs):
		return obs.cpu()

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, **kwargs):
		return self.env.render(**kwargs)

	def _get_obs(self, is_reset=False):
		return

class Pixels(gym.Wrapper):
	def __init__(self, env, cfg, num_frames=3):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, 64, 64), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)

	def _get_visual_obs(self, frame, is_reset=False):
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
			self._frames.append(frame)
		past_n_frames = torch.concatenate(tuple(self._frames), axis=1)
		return past_n_frames

	def reset(self, env_id=None):
		obs = self.env.reset(env_id=env_id)
		return self._get_visual_obs(obs, is_reset=True), {}

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return self._get_visual_obs(obs), reward, terminated, truncated, info

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

    def reset(self, task_idx=None, env_id=None):
        obs = self.env.reset(env_id)
        return self._obs_to_tensor(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.numpy())
        info = defaultdict(float, info)
        return self._obs_to_tensor(obs), reward, terminated, truncated, info

class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(os.path.join(cfg.work_dir, 'eval_video'))
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)
