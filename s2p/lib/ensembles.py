import torch
from torch import nn
from tensordict import from_modules
from copy import deepcopy

"""
Ensemble implementation taken from TD-MPC2
"""

class VectorizedEnsemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to(modules[0].device).to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.modules_list = nn.ModuleList(modules)
		self._n = len(self.modules_list)

	def __len__(self):
		return self._n

	def forward(self, *args, **kwargs):
		return torch.stack(
			[module(*args, **kwargs) for module in self.modules_list],
			dim=0,
		)