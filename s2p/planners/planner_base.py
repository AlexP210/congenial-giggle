import abc
import typing

import torch

from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.policy_model_base import PolicyModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase
from s2p.lib.base import BaseClass

class PlannerBase(abc.ABC, BaseClass):

	# Whether `plan` fills in the per-iteration convergence trace of its info dict. Off by
	# default because collecting the trace costs plan latency, which the control loop pays on
	# every step and `RealTaskEvaluator` turns into a control rate. Turned on from the planner
	# config (`collect_convergence_info: True`), or for the duration of one call by the
	# evaluator which reads it (`PlannerConvergenceEvaluator`).
	collect_convergence_info = False

	@abc.abstractmethod
	def __init__(self, cfg, task):
		super().__init__(cfg)
		self.task = task

	@abc.abstractmethod
	def plan(
			self, 
			dynamics_model:DynamicsModelBase, 
			reward_model:RewardModelBase, 
			value_model:typing.Union[StateValueModelBase, StateActionValueModelBase], 
			policy_model:PolicyModelBase,
			current_state:torch.Tensor, 
			eval_mode:bool, 
			action_prior:torch.Tensor) -> typing.Tuple[torch.Tensor, typing.Dict]:
		pass