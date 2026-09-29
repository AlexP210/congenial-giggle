import abc
import os
import typing
from torch import nn

from s2p.lib.base import BaseClass

class EvaluatorBase(BaseClass, abc.ABC):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.save_path = None

	def set_save_path(self, filepath):
		self.save_path = filepath

	def _save_best_checkpoint(self, model, folder_name:str) -> None:
		"""Write `model` into `<save_path>/<folder_name>`, if a save path has been set."""
		if not self.save_path:
			return
		save_folder = os.path.join(self.save_path, folder_name)
		os.makedirs(save_folder, exist_ok=True)
		model.save_to_folder(save_folder)

	@abc.abstractmethod
	def __call__(self, model) -> typing.Dict[str, typing.Any]:
		pass
