import abc
import typing

from torch.optim import Optimizer

from s2p.models.agent_model import AgentModel
from s2p.losses.loss_function_base import LossFunctionBase
from s2p.lib.base import BaseClass
from s2p.tasks.base.task_base import TaskBase

class TrainerBase(abc.ABC, BaseClass):

    @abc.abstractmethod
    def __init__(self,
                 cfg,
                 task:TaskBase,
                 loss_functions:typing.Dict[str, LossFunctionBase],
                 loss_function_weights:typing.Dict[str, float]
    ):
        super().__init__(cfg)
        self.task = task
        self.loss_functions = loss_functions
        self.loss_function_weights = loss_function_weights
        self._loss_prefixes = {name: f"{name}/" for name in loss_functions}

    def generate_data(self, model:AgentModel) -> None:
        """Collect environment data. No-op for offline trainers."""
        pass

    @abc.abstractmethod
    def fetch_batch(self):
        """Sample a batch from the dataset/buffer. Return None if not ready to train."""
        ...

    @abc.abstractmethod
    def compute_step(self, batch, model:AgentModel, optimizer:Optimizer) -> typing.Dict[str, typing.Any]:
        """Compute losses and perform one gradient step."""
        ...
