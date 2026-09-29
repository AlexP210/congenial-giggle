import abc
import typing

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import tensordict

import gymnasium as gym

from s2p.evaluators.evaluator_base import EvaluatorBase


class OnlineEvaluatorBase(EvaluatorBase):
	def __init__(self, cfg, task):
		super().__init__(cfg)
		self.cfg = cfg
		self.task = task

	@abc.abstractmethod
	def __call__(self, model) -> typing.Dict[str, typing.Any]:
		pass

	@staticmethod
	def _bernoulli_sem(flags:typing.Sequence[bool]) -> float:
		"""Standard error of the mean of a sequence of 0/1 outcomes: sqrt(p(1-p)/n)."""
		n = len(flags)
		if n == 0:
			return float("nan")
		p = float(np.mean(flags))
		return float(np.sqrt(p * (1.0 - p) / n))
