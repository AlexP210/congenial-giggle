import abc
import typing
from collections import defaultdict
import time

import torch
from collections import deque
import numpy as np
import gymnasium as gym
from tqdm import tqdm

from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.planners.planner_base import PlannerBase
from s2p.models.agent_model import AgentModel
from s2p.tasks.base.task_base import TaskBase

def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class PolicyLatencyEvaluator(OnlineEvaluatorBase):
	def __init__(self, cfg, task:TaskBase):
		super().__init__(cfg=cfg, task=task)
		self.cfg=cfg
		self.task = task

	def __call__(self, model:AgentModel, verbose=True) -> typing.Dict[str, typing.Any]:

		# Container for logging info we will output
		info = {}

		# Evaluation loop
		times = []
		iterator = tqdm(range(self.cfg.num_queries), "Policy Latency Evaluation") if verbose else range(self.cfg.num_queries)
		for i in iterator:

			# Initialize the first transition
			# obs_shape, _ = model.get_encoding_function()
			obs = torch.randn(size=self.task.observation_dimension, device=self.cfg.device)

			start = time.time()
			state = model.encoder_model.encode(obs.unsqueeze(0).unsqueeze(0))
			_ = model.act(state)
			times.append(time.time() - start)


		info.update({
			f"policy_latency": np.mean(times),
			f"policy_latency_sem": np.std(times)/np.sqrt(len(times)),
		})

		return info