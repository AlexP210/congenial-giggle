import abc
import typing

import torch

from s2p.models.base.policy_model_base import PolicyModelBase

class ObservationPolicyModelBase(PolicyModelBase):
	"""
	Base class representing a model that can be used for planning

	`PolicyModelBase` is already an `abc.ABC`, so listing `abc.ABC` here as well is what
	linearizes inconsistently -- a base cannot precede its own subclass in the MRO. The
	abstractness is inherited with it, `ABCMeta` included, so `abstractmethod` below still
	bites.
	"""

	@abc.abstractmethod
	def policy(self, o) -> torch.Tensor:
		pass
