import abc

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from s2p.tasks.base.task_base import TaskBase

class OnlineTaskBase(TaskBase):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.cfg = cfg


	@abc.abstractmethod
	def make_env(self, num_envs: int = None, max_episode_steps: int = None) -> gym.Env:
		"""
		An env stepping `num_envs` copies of the task in parallel.

		Every env this project hands out is *batched*: observations, `reward`, `terminated` and
		`truncated` all carry a leading env axis of size `num_envs`, and `step` takes an action of
		shape `(num_envs, *action_dimension)` -- at `num_envs=1` as much as at 16. One layout for
		every task family is what lets the collection and evaluation loops have a single code
		path, rather than a squeeze here and a special case there.

		`None` means the task's own configured default. A task whose simulator cannot run copies
		in parallel (dm_control) accepts the argument and raises for anything above 1, rather than
		ignoring it and quietly serving one env to a caller that asked for eight.

		`max_episode_steps` overrides the time limit this env truncates at, for a component that
		wants to score on shorter (or longer) episodes than the task's own default -- an evaluator
		mostly, which is where the knob is configured. `None` keeps the task's default. The unit is
		the task's own: *primitive* env steps on ManiSkill, where `frame_skip` then makes an
		episode `max_episode_steps / frame_skip` macro steps long, so the number means the same
		thing as the task's registered limit and does not move when `frame_skip` does.

		It changes only the env handed back, not `task.episode_length`, which describes the task
		itself and is what a model sizing itself to the task (`TDMPC2WorldModel`) reads.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def get_control_interval(self) -> float:
		raise NotImplementedError

	@staticmethod
	def _dims_from_space(space: gym.Space):
		if isinstance(space, gym.spaces.Dict):
			return {key: subspace.shape for key, subspace in space.items()}
		return space.shape

	@classmethod
	def _per_env_dims_from_space(cls, space: gym.Space):
		"""
		`_dims_from_space` with the env axis dropped, for `TaskBase.action_dimension` and
		`observation_dimension`.

		Those two describe *one* env's action and observation -- they are what a model's layers
		are sized against, what an encoder measures its leading batch dims from
		(`batch_dims = obs.shape[:-len(obs_dim)]`), and what the offline datasets are
		cross-checked against, none of which have anything to do with how many copies of the env
		happen to be stepping. Since `make_env` batches every env, reading the spaces directly
		would fold `num_envs` into the feature shape: an encoder would then treat the env axis as
		part of the observation, and the planner would build its action tensors with an extra
		axis in the wrong place. Neither raises -- they just compute the wrong thing.
		"""
		dims = cls._dims_from_space(space)
		if isinstance(dims, dict):
			return {key: shape[1:] for key, shape in dims.items()}
		return dims[1:]

	@classmethod
	def _per_env_limits_from_space(cls, space: gym.Space):
		"""`_limits_from_space` for one env, i.e. env 0's slice of a batched space.

		Every env in a batch is the same task under the same controller, so they share one set of
		bounds; taking env 0's keeps `action_limits` broadcastable against a proposal of any rank
		(see `_limits_from_space`), which a `(num_envs, *action_dimension)` pair would not be.
		"""
		low, high = cls._limits_from_space(space)
		return low[0], high[0]

	@staticmethod
	def _limits_from_space(space: gym.Space):
		"""
		The bounds a space declares, as `(low, high)` arrays, for `TaskBase.action_limits`.

		Kept at the space's own shape rather than reduced to scalars, so that per-component
		bounds survive and so the pair broadcasts against a batch of proposed actions of any
		rank. Anything other than a Box declares no bounds to read and reports infinities --
		clamping against those is a no-op, which is the honest answer rather than a silently
		invented restriction.
		"""
		if isinstance(space, gym.spaces.Box):
			return (
				np.asarray(space.low, dtype=np.float32),
				np.asarray(space.high, dtype=np.float32),
			)
		shape = () if space is None or space.shape is None else space.shape
		return (
			np.full(shape, -np.inf, dtype=np.float32),
			np.full(shape, np.inf, dtype=np.float32),
		)
