import typing
from collections import deque

import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from omegaconf import OmegaConf
from tensordict import TensorDict

from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.evaluators.evaluator_group import EvaluatorGroup
from s2p.lib.transition_data import OnlineTransitionDataset
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.models.agent_model import AgentModel
from s2p.lib.utils import episode_boundary


class OnlineWrapper(OnlineEvaluatorBase):
	"""
	Adapts one or more `OfflineEvaluator`s into a single `OnlineEvaluator`.

	Rolls the agent out in the environment for `cfg.num_transitions` transitions, chops the
	resulting trajectories into subtrajectories of length `cfg.horizon` (as the online
	trainer does), and runs the wrapped offline evaluators over that freshly collected,
	temporary dataset. This lets dataset-based metrics (losses, rollout error, t-SNE, ...)
	be measured on on-policy data instead of a fixed offline dataset.

	`evaluators` is a list, a name -> evaluator mapping, or a single evaluator; see
	`EvaluatorGroup`, which also settles how their outputs are named. Several of them share
	the one rollout, which is the point of taking more than one: the collection is by far the
	expensive part -- `num_transitions` planned steps, against a dataset pass per evaluator --
	and a wrapper per evaluator would also hold an environment per evaluator.

	Image observations are stored as uint8 (as the offline datasets hold them) so that the
	temporary buffer costs a byte per channel rather than four; encoders normalize them
	themselves. Set `cfg.store_images_as_uint8: False` to keep whatever dtype the env emits.
	"""

	def __init__(self, cfg:OmegaConf, task:OnlineTaskBase, evaluators):

		super().__init__(cfg, task)
		self.cfg = cfg

		self.task = task
		self.evaluators = EvaluatorGroup(evaluators)

		# Episodes run `num_envs` at a time, in lockstep; see `generate_episode`.
		self.num_envs = int(self.cfg.num_envs)
		self.env = task.make_env(num_envs=self.num_envs)

		# Image observations are stored as uint8 (1 byte per channel instead of 4). An
		# observation whose space declares a uint8 dtype is an image - the same signal
		# `ManiSkillWrapper` uses, since shape is not reliable (e.g. dino patch features
		# are also rank-3 but are not images).
		observation_space = getattr(self.env, "observation_space", None)
		if isinstance(observation_space, gym.spaces.Dict):
			self.image_keys = {
				key for key, space in observation_space.spaces.items()
				if space.dtype == np.uint8
			}
			self.observation_is_image = False
		else:
			self.image_keys = None
			self.observation_is_image = (
				observation_space is not None and observation_space.dtype == np.uint8
			)
		self._checked_image_range = False

		# `num_transitions` is a total across the envs, and the shortest useful rollout is one
		# window on every env at once, so that is what has to fit.
		if self.num_transitions < self.horizon * self.num_envs:
			raise ValueError(
				f"num_transitions ({self.num_transitions}) is smaller than horizon ({self.horizon}) "
				f"x num_envs ({self.num_envs}), so not a single subtrajectory can be collected."
			)

	def set_save_path(self, filepath):
		# Forward the save path so the wrapped evaluators can still write their own artifacts
		super().set_save_path(filepath)
		self.evaluators.set_save_path(filepath)

	@property
	def horizon(self) -> int:
		return int(self.cfg.horizon)

	@property
	def num_transitions(self) -> int:
		"""Total number of transitions to collect, counting the one produced by each reset."""
		return int(self.cfg.num_transitions)

	@property
	def stride(self) -> int:
		"""Number of steps between the start of consecutive subtrajectories."""
		overlap_ratio = float(self.cfg.get("overlap_ratio", 0.0))
		return max(1, self.horizon - int(self.horizon * overlap_ratio))

	@property
	def capacity(self) -> int:
		"""
		Upper bound on the number of subtrajectories `num_transitions` transitions can yield.

		The first subtrajectory of an episode costs `horizon` transitions and every later one
		costs a further `stride`, so a budget of `n` transitions can never yield more than
		`n // stride` subtrajectories (exactly `n // horizon` when they do not overlap).

		Independent of `num_envs`: a step over `num_envs` envs costs `num_envs` transitions and
		emits `num_envs` windows, so the ratio the bound rests on is unchanged.
		"""
		return max(1, self.num_transitions // self.stride)

	def _cast_image(self, image:torch.Tensor) -> torch.Tensor:
		"""Cast a single image observation to uint8, leaving it alone if it already is."""
		if not image.is_floating_point():
			return image

		# Encoders take images in [0, 255] and normalize internally (see
		# DeterministicCNNEncoderModel.encode, DINOV3EncoderModel's ToDtype(scale=True)),
		# so complain rather than silently quantizing [0, 1] images down to zeros
		if not self._checked_image_range:
			self._checked_image_range = True
			if float(image.max()) <= 1.0:
				print(
					"[OnlineWrapper] Image observations look normalized to [0, 1], but they are "
					"expected in [0, 255]; storing them as uint8 will quantize them to zeros. "
					"Set `store_images_as_uint8: False` if that is really the observation scale."
				)

		return image.round().clamp(0, 255).to(torch.uint8)

	def cast_observation(self, obs):
		"""Cast the image entries of an observation to uint8 so the buffer holds bytes, not floats."""
		if not self.cfg.get("store_images_as_uint8", True):
			return obs

		# Flat observation: either the whole thing is an image, or none of it is
		if self.image_keys is None:
			return self._cast_image(obs) if self.observation_is_image else obs

		# Composite observation: only the image keys are cast
		if not self.image_keys:
			return obs
		cast = {
			key: self._cast_image(value) if key in self.image_keys else value
			for key, value in obs.items()
		}
		if isinstance(obs, TensorDict):
			return TensorDict(cast, batch_size=obs.batch_size, device=obs.device)
		return type(obs)(cast)

	def _new_subtrajectory(self, obs:torch.Tensor) -> typing.Dict[str, typing.Deque]:
		"""
		Initialize the rolling subtrajectory window from a freshly reset environment.

		Follows the online trainer's convention: `action[t]`, `reward[t]` are the action
		which produced `obs[t]` and the reward it earned, so the entry belonging to the
		reset observation has no action/reward and is filled with NaN (loss functions
		drop index 0 of those keys).
		"""
		window = {
			"obs": deque([obs.detach().to(self.cfg.load_device)], maxlen=self.horizon),
			"action": deque([torch.full_like(self.env.rand_act(), float('nan')).to(self.cfg.load_device)], maxlen=self.horizon),
			"reward": deque([torch.full((self.num_envs, 1), float("nan"), dtype=torch.float32, device=self.cfg.load_device)], maxlen=self.horizon),
			"terminated": deque([torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.cfg.load_device)], maxlen=self.horizon),
			"truncated": deque([torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.cfg.load_device)], maxlen=self.horizon),
		}
		return window

	def _window_to_tensordict(self, window:typing.Dict[str, typing.Deque]) -> TensorDict:
		return TensorDict(
			source={
				key: torch.stack(list(values), dim=0)
				for key, values in window.items()
			},
			batch_size=[self.horizon, self.num_envs],
		)

	def generate_action(self, model:AgentModel, obs:torch.Tensor, previous_plan:torch.Tensor):
		"""Generate an action for the current observation, returning it with the updated plan."""
		# (T=1, B, ...): the observation already carries the env axis, so only time is added.
		state = model.encoder_model.encode(obs.unsqueeze(0))
		if self.cfg.action_mode == "plan":
			if previous_plan is None:
				previous_plan = torch.zeros(
					size=(model.planner.cfg.horizon, self.num_envs, *self.task.action_dimension),
					device=self.cfg.batch_device
				)
			action_prior = torch.zeros_like(previous_plan)
			action_prior[:-1] = previous_plan[1:]
			plan, _ = model.plan(state, action_prior)
			return plan[0], plan
		elif self.cfg.action_mode == "act":
			return model.act(state).squeeze(0), previous_plan
		else:
			raise ValueError("action_mode must be one of ('plan', 'act').")

	@torch.no_grad()
	def generate_episode(self, model:AgentModel, dataset:OnlineTransitionDataset, budget:int):
		"""
		Roll out one episode on every env at once, adding every subtrajectory of length `horizon`
		(taken every `stride` steps) to `dataset`. The rollout stops at the end of the episode or
		once `budget` transitions have been collected, whichever comes first.

		The envs step in lockstep and are reset together, so a step costs `num_envs` transitions
		and emits `num_envs` windows -- one per env, none of which straddles a reset.

		Returns the per-env episode returns, the number of transitions collected, and whether the
		episode ran to completion (the returns are only meaningful if it did).
		"""
		# Cast before acting as well as before storing, so the dataset holds exactly the
		# observations the agent conditioned on
		obs = self.cast_observation(self.env.reset().to(self.cfg.batch_device))
		window = self._new_subtrajectory(obs)
		previous_plan = None

		done = False
		return_ = torch.zeros(self.num_envs, device=self.cfg.batch_device)
		# Index of the most recently appended step (0 is the reset step), so `(t+1) * num_envs`
		# transitions of the budget have been consumed -- every env contributes one per step.
		t = 0

		while not done and (t + 1) * self.num_envs < budget:

			# Generate an action & step the environment
			action, previous_plan = self.generate_action(model, obs, previous_plan)
			obs, reward, terminated, truncated, _ = self.env.step(action.cpu())
			obs = self.cast_observation(obs.to(self.cfg.batch_device))
			done = episode_boundary(terminated, truncated)

			# Add the transition to the rolling window. Everything the env reports is already
			# per-env, so these only need the trailing feature axis the buffer stores them with.
			window["obs"].append(obs.detach().to(self.cfg.load_device))
			window["action"].append(action.detach().float().to(self.cfg.load_device))
			window["reward"].append(reward.to(dtype=torch.float32, device=self.cfg.load_device).reshape(self.num_envs, 1))
			window["terminated"].append(terminated.to(dtype=torch.bool, device=self.cfg.load_device).reshape(self.num_envs, 1))
			window["truncated"].append(truncated.to(dtype=torch.bool, device=self.cfg.load_device).reshape(self.num_envs, 1))

			return_ += reward.to(self.cfg.batch_device).reshape(self.num_envs)
			t += 1

			# Once the window is full, emit it every `stride` steps so that
			# subtrajectories never straddle an episode boundary
			start = t - self.horizon + 1
			if start >= 0 and start % self.stride == 0:
				dataset.add_subtrajectories(self._window_to_tensordict(window))

		return return_.tolist(), (t + 1) * self.num_envs, bool(done)

	@torch.no_grad()
	def generate_dataset(self, model:AgentModel) -> OnlineTransitionDataset:
		"""Collect `cfg.num_transitions` transitions of on-policy data into a fresh dataset."""
		dataset = OnlineTransitionDataset(
			capacity=self.capacity,
			horizon=self.horizon,
			load_device=self.cfg.load_device,
			batch_device=self.cfg.batch_device,
		)

		# Returns of the episodes which ran to completion, for logging
		returns = []

		remaining = self.num_transitions
		progress = tqdm(total=self.num_transitions, desc="Online Data Collection")
		# Stop once the remaining budget is too small to hold another subtrajectory,
		# rather than resetting the environment for data we would have to throw away
		while remaining >= self.horizon * self.num_envs:
			episode_returns, collected, completed = self.generate_episode(model, dataset, remaining)
			remaining -= collected
			progress.update(collected)
			if completed:
				returns.extend(episode_returns)
		progress.close()

		return dataset, returns

	def __call__(self, model:AgentModel) -> typing.Dict[str, typing.Any]:

		# Collect the temporary on-policy dataset, once for all of the wrapped evaluators
		dataset, returns = self.generate_dataset(model)

		# Evaluate over it with each wrapped offline evaluator
		info = self.evaluators(model, dataset)

		return info
