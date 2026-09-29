import typing
from collections.abc import Iterable
import os
from datetime import datetime
from omegaconf import OmegaConf

import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch

from s2p.runners.runner_base import RunnerBase
from s2p.models.agent_model import AgentModel
from s2p.tasks.base.task_base import TaskBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.evaluators.evaluator_base import EvaluatorBase
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase


class EvaluationRunner(RunnerBase):
	def __init__(self, cfg:OmegaConf, task:TaskBase, model:AgentModel, evaluators:typing.Dict[str, EvaluatorBase]=None):

		super().__init__(cfg, model, task)
		self.cfg = cfg

		if evaluators is None: 
			raise ValueError("Provided no evaluators to EvaluationRunner.")
		else:
			self.evaluators = evaluators

		# Check if any offline evaluators provided
		offline_evaluator_requested = any([isinstance(e, OfflineEvaluatorBase) for e in self.evaluators.values()])
		# If so, then make sure that we have an offline task (otherwise no data)
		task_is_offline = isinstance(task, OfflineTaskBase)
		
		self.evaluation_dataset = None
		if offline_evaluator_requested and not task_is_offline:
			raise ValueError("Offline evaluator requested, but task is online - we do not have data to evaluate over.")
		elif offline_evaluator_requested and task_is_offline:
			self.evaluation_dataset, _ = task.make_dataset()


		self.model.requires_grad_(False)
		self.model.eval()

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.checkpoint_folders = {}
		if self.cfg.save_path is not None:
			self.save_path = Path(self.cfg.save_path) / f"{self.cfg.run_name}-{timestamp}" 
			if not os.path.exists(self.save_path): os.mkdir(self.save_path)
			else: raise ValueError(f"Run name {self.cfg.run_name} already exists at {self.cfg.save_path}.")

			# # Create the checkpoint folders
			# for evaluator_name, evaluator in self.evaluators.items():
			# 	evaluator_save_path = os.path.join(self.save_path, evaluator_name)
			# 	if not os.path.exists(evaluator_save_path):
			# 		os.mkdir(evaluator_save_path)
			# 	evaluator.set_save_path(evaluator_save_path)

	def run(self):
		if self.cfg.use_wandb:
			if self.full_cfg is None:
				raise ValueError("With `use_wandb=True`, must call `RunnerBase.log_cfg()` before `RunnerBase.run()`")
			self.wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
				group=self.cfg.group_name,
				job_type="eval",
                config=OmegaConf.to_container(self.full_cfg)
            )
		
		evaluation_info = {}

		self.model.eval()
		self.model.requires_grad_(False)
		
		for evaluator_name, evaluator in tqdm(self.evaluators.items(), desc="Evaluation"):
			if isinstance(evaluator, OfflineEvaluatorBase):
				evaluator_output = evaluator(self.model, self.evaluation_dataset)
			elif isinstance(evaluator, OnlineEvaluatorBase):
				evaluator_output = evaluator(self.model)
			for key, value in evaluator_output.items():
				key_with_evaluator_name = f"{evaluator_name}/{key}"
				evaluation_info[key_with_evaluator_name] = value
		self._log_wandb(evaluation_info, ["evaluation",])
		wandb.finish()

	def _log_wandb(self, info, groups):
		if not self.cfg.use_wandb: return

		wandb_data = {}
		for key, value in info.items():
			new_key = f"{'/'.join(groups)}/{key}"
			if isinstance(value, (int, float, np.number)):
				wandb_data[new_key] = value
			elif value.ndim==1:
				wandb_data[new_key] = wandb.Histogram(value)
			elif value.ndim==3:
				wandb_data[new_key] = wandb.Image(value)
			elif value.ndim==4:
				wandb_data[new_key] = wandb.Video(value, fps=15, format="mp4")
		
		self.wandb_run.log(
			data=wandb_data
		)