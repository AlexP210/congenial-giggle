import abc
import typing

import torch

from s2p.lib.base import BaseClass

class EncoderModelBase(abc.ABC, BaseClass):
	"""
	Base class representing a module which can encode observations.
	"""

	def __init__(self, cfg):
		super().__init__(cfg)

	@abc.abstractmethod
	def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None):
		pass

	@abc.abstractmethod
	def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
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