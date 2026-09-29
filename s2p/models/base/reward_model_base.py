import abc
import typing

import torch

from s2p.lib.base import BaseClass

class RewardModelBase(abc.ABC, BaseClass):
	"""
	Base class representing a model that can be used for planning
	"""

	def __init__(self, cfg):
		super().__init__(cfg)

	@abc.abstractmethod
	def reward(self, s, a) -> torch.Tensor:
		pass

	@abc.abstractmethod
	def get_reward_function(self) -> typing.Tuple[typing.Tuple[int], torch.Tensor]:
		pass

	@abc.abstractmethod
	def save_to_file(self, filepath) -> None:
		pass

	@abc.abstractmethod
	def load_from_file(self, path):
		pass

	def on_parameter_update_callback(self) -> None:
		"""
		Called on parameter updates, use if needed (ex. to maintain EMA of weights)
		"""
		pass