import contextlib
import typing

import torch

from s2p.models.base.encoder_model_base import EncoderModelBase

class EMATargetEncoderBase(EncoderModelBase):
	"""
	Mixin for encoders that maintain a detached exponential moving average (EMA)
	of their own parameters and expose `encode_target`, which runs `encode` using
	those EMA weights (a slow-moving "target" encoder).

	Intended to be combined with `torch.nn.Module`, since the EMA is built from
	`named_parameters()`, e.g.:

		class MyEncoder(EMATargetEncoderBase, torch.nn.Module):
			...

	The EMA decay (momentum) is read from `cfg.ema_decay`, defaulting to 0.99.
	A higher decay yields a slower-moving target.
	"""

	def __init__(self, cfg):
		super().__init__(cfg)
		self._ema_decay = cfg.ema_decay
		# Populated lazily on the first parameter-update callback, since the
		# concrete subclass typically builds its parameters *after* super().__init__.
		self._ema_parameters: typing.Dict[str, torch.Tensor] = {}

	def _initialize_ema_parameters(self) -> None:
		self._ema_parameters = {
			name: parameter.detach().clone()
			for name, parameter in self.named_parameters()
		}

	def on_parameter_update_callback(self) -> None:
		"""
		Update the EMA parameters towards the current (online) parameters:

			ema <- decay * ema + (1 - decay) * parameter
		"""
		if not self._ema_parameters:
			self._initialize_ema_parameters()
			return

		with torch.no_grad():
			for name, parameter in self.named_parameters():
				self._ema_parameters[name].lerp_(parameter.detach(), 1.0 - self._ema_decay)

	@contextlib.contextmanager
	def _use_ema_parameters(self):
		"""Temporarily swap the online parameters for their EMA counterparts."""
		if not self._ema_parameters:
			self._initialize_ema_parameters()

		saved_data: typing.Dict[str, torch.Tensor] = {}
		try:
			for name, parameter in self.named_parameters():
				saved_data[name] = parameter.data
				parameter.data = self._ema_parameters[name]
			yield
		finally:
			for name, parameter in self.named_parameters():
				parameter.data = saved_data[name]

	def encode_target(
		self,
		observation: torch.Tensor,
		previous_state: torch.Tensor = None,
		action: torch.Tensor = None,
	) -> torch.Tensor:
		"""Identical to `encode`, but evaluated using the EMA parameters."""
		with torch.no_grad(), self._use_ema_parameters():
			return self.encode(observation, previous_state=previous_state, action=action)

	def encode_target_distribution(self, observation: torch.Tensor):
		"""
		Identical to `encode_distribution`, but evaluated using the EMA parameters.

		A caller wanting a *sampled* target has to come through here rather than through
		`encode_target`: on a stochastic encoder `encode` follows `cfg.sample_mean`, which
		is a statement about what consumers (the planner, the value loss) should be handed,
		not about what a loss's target should be. Only defined for encoders that have a
		distribution to expose; a deterministic one raises `AttributeError`, which is the
		honest answer.
		"""
		with torch.no_grad(), self._use_ema_parameters():
			return self.encode_distribution(observation)
