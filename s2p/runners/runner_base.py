import abc

from omegaconf import OmegaConf

from s2p.models.agent_model import AgentModel
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.base import BaseClass

class RunnerBase(abc.ABC, BaseClass):

	def __init__(self, cfg:OmegaConf, model:AgentModel, task:TaskBase):
		super().__init__(cfg)
		# Deliberately no seeding here. `model` and `task` arrive already constructed -- Hydra
		# instantiates a config's dependencies before the object declaring them -- so seeding at
		# this point is too late to make initialization reproducible, and seeding here anyway
		# would only make an unseeded run *look* seeded. `main.py` calls `seed_all` before
		# `instantiate`, which is the only point that precedes every consumer. See
		# s2p/lib/seeding.py.
		self.model = model
		self.model.to(cfg.device)

		self.task = task

		self.full_cfg = None
	
	@abc.abstractmethod
	def run(self): 
		pass

	def log_cfg(self, full_cfg:OmegaConf):
		self.full_cfg = full_cfg