import typing
from collections.abc import Iterable
import gc
import os
from datetime import datetime
import numbers

from omegaconf import OmegaConf
import wandb
from tqdm import tqdm
import torch
from pathlib import Path
import numpy as np

from s2p.runners.runner_base import RunnerBase
from s2p.trainers.trainer_base import TrainerBase
from s2p.models.agent_model import AgentModel
from s2p.evaluators.evaluator_base import EvaluatorBase
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.tasks.base.task_base import TaskBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase


class TrainingRunner(RunnerBase):

	def __init__(self, cfg:OmegaConf, task:TaskBase, model:AgentModel, trainer:TrainerBase, evaluators:typing.Dict[str, EvaluatorBase]=None):
		# Send the required info to the base class
		super().__init__(cfg, model, task)
		self.cfg = cfg

		# Set up for training
		self.trainer = trainer
		self.optimizer = torch.optim.AdamW(
			lr=self.cfg.learning_rate,
			params=self.model.parameters()
		)

		# Set up for evaluations
		if evaluators is None: 
			self.evaluators = {}
		else:
			self.evaluators = evaluators
		
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.save_path = Path(self.cfg.save_path) / f"{self.cfg.run_name}-{timestamp}" 
		if not os.path.exists(self.save_path): os.mkdir(self.save_path)
		else: raise ValueError(f"Run name {self.cfg.run_name} already exists at {self.cfg.save_path}.")

		# Create the checkpoint folders
		self.checkpoint_folders = {}
		for evaluator_name, evaluator in self.evaluators.items():
			evaluator_save_path = os.path.join(self.save_path, evaluator_name)
			if not os.path.exists(evaluator_save_path):
				os.mkdir(evaluator_save_path)
			evaluator.set_save_path(evaluator_save_path)

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
		
		self.model.train()
		self.model.requires_grad_(True)

		gc.disable()
		gc.collect()

		if self.cfg.compile:
			def _step(batch):
				return self.trainer.compute_step(batch, self.model, self.optimizer)
			compiled_step = torch.compile(_step, mode=self.cfg.compile_mode)
		else:
			compiled_step = lambda batch: self.trainer.compute_step(batch, self.model, self.optimizer)

		training_bar = tqdm(iterable=range(self.cfg.max_iterations), desc="Training")
		for iteration in training_bar:

			while True: # Generate data as long as we don't have enough to sample
				self.trainer.generate_data(self.model)
				batch = self.trainer.fetch_batch()
				if batch is not None: break
			training_info = compiled_step(batch)

			if iteration % self.cfg.log_interval == 0:
				gc.collect()
				self._log_wandb(info=training_info, iteration=iteration, groups=["train"])

			if iteration % self.cfg.evaluation_interval == 0:
				evaluation_info = {}
				self.model.requires_grad_(False)
				self.model.eval()
				for evaluator_name, evaluator in self.evaluators.items():
					if isinstance(evaluator, OfflineEvaluatorBase):
						evaluator_output = evaluator(self.model, self.trainer.validation_dataset)
					else:
						evaluator_output = evaluator(self.model)
					
					evaluation_info.update({ # Label the evaluator output with the evaluator name
						f"{evaluator_name}/{key}": value 
						for key, value in evaluator_output.items()
					})
					gc.collect()
				self._log_wandb(info=evaluation_info, iteration=iteration, groups=["evaluation"])
				self.model.requires_grad_(True)
				self.model.train()
				# torch.compiler.cudagraph_mark_step_begin()
				
		self.wandb_run.finish()

	def _log_wandb(self, info, iteration, groups):
		if not self.cfg.use_wandb: return

		wandb_data = {}
		for key, value in info.items():
			new_key = f"{'/'.join(groups)}/{key}"
			if isinstance(value, numbers.Number):
				wandb_data[new_key] = value
			elif value.ndim==0:
				wandb_data[new_key] = value.item()
			elif value.ndim==1:
				wandb_data[new_key] = wandb.Histogram(value)
			elif value.ndim==3:
				wandb_data[new_key] = wandb.Image(value)
			elif value.ndim==4:
				wandb_data[new_key] = wandb.Video(value, fps=15, format="mp4")
		
		self.wandb_run.log(
			data=wandb_data,
			step=iteration
		)