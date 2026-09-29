import abc

from omegaconf import OmegaConf

from s2p.lib.base import BaseClass

class TaskBase(BaseClass):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.cfg = cfg
		self._action_dimension = None
		self._action_limits = None
		self._observation_dimension = None
		self._task_name = None

	@property
	def task_name(self) -> str:
		if self._task_name is None:
			raise NotImplementedError("Subclass must set task_name")
		return self._task_name

	@property
	def action_dimension(self) -> int:
		if self._action_dimension is None:
			raise NotImplementedError("Subclass must set action_dimension")
		return self._action_dimension
	
	@action_dimension.setter
	def action_dimension(self, value):
		self._action_dimension = value

	@property
	def action_limits(self):
		"""
		The bounds an action is required to respect, as `(low, high)` broadcastable against
		`action_dimension`.

		A statement about what the environment will *accept*, not about where good actions
		lie: the planners clamp their proposals to this so the action they score is the
		action that gets executed. The families this codebase wraps differ in kind --
		ManiSkill's controller clips to its own box, dm_control neither clips nor validates
		(an out-of-spec control reaches MuJoCo verbatim) -- so a task reports what it knows
		and reports infinities where nothing is enforced.
		"""
		if self._action_limits is None:
			raise NotImplementedError("Subclass must set action_limits")
		return self._action_limits

	@action_limits.setter
	def action_limits(self, value):
		self._action_limits = value

	@property
	def observation_dimension(self) -> int:
		if self._observation_dimension is None:
			raise NotImplementedError("Subclass must set observation_dimension")
		return self._observation_dimension
	
	@observation_dimension.setter
	def observation_dimension(self, value):
		self._observation_dimension = value

	@property
	def episode_length(self) -> int:
		if self._episode_length is None:
			raise NotImplementedError("Subclass must set observation_dimension")
		return self._episode_length
	
	@episode_length.setter
	def episode_length(self, value):
		self._episode_length = value

	def sample_goal_observation(self):
		return
