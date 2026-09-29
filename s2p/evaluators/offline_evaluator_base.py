import abc
import typing

from torch.utils.data import Dataset, DataLoader
from torch import nn
import tensordict

from s2p.evaluators.evaluator_base import EvaluatorBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase


class OfflineEvaluatorBase(EvaluatorBase):

	def __init__(self, cfg, task:OfflineTaskBase):
		super().__init__(cfg)
		# Load the dataset
		self.task = task

	@abc.abstractmethod
	def __call__(self, model, dataset) -> typing.Dict[str, typing.Any]:
		pass
