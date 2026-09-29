import typing
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, RandomSampler, random_split
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
import tqdm

from torch.utils.data import DataLoader, Subset

from s2p.losses.loss_function_base import LossFunctionBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.lib.utils import collate_batch

class LossEvaluator(OfflineEvaluatorBase):

	def __init__(self, cfg, 
			  task:OfflineTaskBase, 
			  loss_functions:typing.Dict[str, LossFunctionBase], 
			  loss_function_weights:typing.Dict[str, float]
		):

		super().__init__(cfg, task)
		self.cfg = cfg
		self.loss_functions = loss_functions
		self.loss_function_weights = loss_function_weights
		self.best_loss = float("inf")
		# Best value seen for each component loss on its own, tracked separately from the
		# weighted total: the checkpoint that is best overall is not necessarily the one that
		# is best at, say, distillation.
		self.best_component_losses = {name: float("inf") for name in self.loss_functions}

	def __call__(self, model, dataset) -> typing.Dict[str, typing.Any]:
		info = {}

		n = len(dataset)
		k = int(n * self.cfg.fraction)  # e.g. fraction=0.1 for 10%
		indices = random.sample(range(n), k)

		data_loader = DataLoader(
			dataset=Subset(dataset, indices),
			batch_size=self.cfg.batch_size,
			shuffle=True,
		    collate_fn=collate_batch
		)

		total_loss = 0
		for batch in tqdm.tqdm(data_loader, "Loss Evaluation"):
			# The dataset builds samples on the CPU (see the task's `batch_device`), so the
			# batch has to be placed before it reaches a model living on the GPU. A no-op for
			# a task whose dataset still places its own samples.
			batch = torch.permute(batch.to(self.cfg.device), dims=(1, 0))
			for loss_function_name, loss_function in self.loss_functions.items():
				loss = loss_function(batch, model)[0]
				if loss_function_name in info: info[loss_function_name] += loss.item()/len(data_loader)
				else: info[loss_function_name] = loss.item()/len(data_loader)
				total_loss += self.loss_function_weights[loss_function_name] * loss / len(data_loader)
		info["total_loss"] = total_loss.item()

		if total_loss.item() < self.best_loss:
			self.best_loss = total_loss.item()
			self._save_best_checkpoint(model, "best_total_loss")

		for loss_function_name in self.loss_functions:
			component_loss = info[loss_function_name]
			if component_loss < self.best_component_losses[loss_function_name]:
				self.best_component_losses[loss_function_name] = component_loss
				self._save_best_checkpoint(model, f"best_{loss_function_name}_loss")

		return info
