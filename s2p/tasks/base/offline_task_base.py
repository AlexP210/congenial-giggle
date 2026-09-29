import abc

from omegaconf import OmegaConf
import torch

from s2p.tasks.base.task_base import TaskBase

class OfflineTaskBase(TaskBase):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.cfg = cfg
		self._training_dataset = None
		self._validation_dataset = None
	
	@property
	def training_dataset(self) -> torch.utils.data.Dataset:
		if self._training_dataset is None:
			self._training_dataset, self._validation_dataset = self.make_dataset()
		return self._training_dataset
	
	@property
	def validation_dataset(self) -> torch.utils.data.Dataset:
		if self._validation_dataset is None:
			self._training_dataset, self._validation_dataset = self.make_dataset()
		return self._validation_dataset
	
	@abc.abstractmethod
	def make_dataset(self) -> torch.utils.data.Dataset:
		raise NotImplementedError

	@staticmethod
	def _dims_from_dataset(dataset: torch.utils.data.Dataset):
		sample = dataset[0]
		obs, action = sample["obs"], sample["action"]
		if hasattr(obs, "keys"):
			observation_dimension = {key: tuple(obs[key].shape[1:]) for key in obs.keys()}
		else:
			observation_dimension = tuple(obs.shape[1:])
		action_dimension = tuple(action.shape[1:])
		return observation_dimension, action_dimension
