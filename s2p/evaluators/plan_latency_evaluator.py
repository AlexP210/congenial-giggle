import abc
import typing
from collections import defaultdict
import time

import torch
from collections import deque
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from tensordict import TensorDict

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


class PlanLatencyEvaluator(OnlineEvaluatorBase):
	def __init__(self, cfg, task:TaskBase):
		super().__init__(cfg=cfg, task=task)
		self.cfg=cfg
		self.task = task

	def __call__(self, model:AgentModel, verbose=True) -> typing.Dict[str, typing.Any]:

		# Container for logging info we will output
		info = {}

		# Peak GPU memory is read off PyTorch's caching allocator, so it covers tensors
		# (weights, the planner's population, activations) but not the CUDA context itself.
		track_memory = torch.device(self.cfg.device).type == "cuda"
		if track_memory:
			torch.cuda.synchronize(self.cfg.device)
			baseline_allocated = torch.cuda.memory_allocated(self.cfg.device)

		# Evaluation loop
		times = []
		peak_allocated = []
		peak_reserved = []
		iterator = tqdm(range(self.cfg.num_plans), "Plan Latency Evaluation") if verbose else range(self.cfg.num_plans)
		for i in iterator:

			# Initialize the first transition
			# obs_shape, _ = model.get_encoding_function()
			# Get the correct observation dimension
			obs_dim = self.task.observation_dimension
			if isinstance(obs_dim, dict):
				obs = TensorDict({
					k: torch.randn(size=obs_dim[k], device=self.cfg.device)
					for k in obs_dim.keys()
				}, device=self.cfg.device)			
			else:
				obs = torch.randn(size=obs_dim, device=self.cfg.device)

			action_prior = torch.randn(size=(model.planner.cfg.horizon, self.task.action_dimension[-1]), device=self.cfg.device)

			# Drain the queue before starting the clock and again before stopping it: CUDA
			# launches are asynchronous, so without the second sync the timer would stop
			# while the plan's kernels were still running.
			if track_memory:
				torch.cuda.synchronize(self.cfg.device)
				torch.cuda.reset_peak_memory_stats(self.cfg.device)
			start = time.time()
			state = model.encoder_model.encode(obs.unsqueeze(0).unsqueeze(0))
			_, _ = model.plan(state, action_prior)
			if track_memory:
				torch.cuda.synchronize(self.cfg.device)
			times.append(time.time() - start)
			if track_memory:
				peak_allocated.append(torch.cuda.max_memory_allocated(self.cfg.device))
				peak_reserved.append(torch.cuda.max_memory_reserved(self.cfg.device))


		info.update({
			f"plan_latency": np.mean(times),
			f"plan_latency_sem": np.std(times)/np.sqrt(len(times)),
			f"plan_latencies": np.array(times, dtype=np.float32),
		})
		if track_memory:
			# Bytes. `peak_allocated - baseline_allocated` is what one plan needs on top of
			# the loaded model; `peak_reserved` includes the allocator's cache.
			info.update({
				"plan_memory_baseline_allocated": baseline_allocated,
				"plan_memory_peak_allocated": np.array(peak_allocated, dtype=np.int64),
				"plan_memory_peak_reserved": np.array(peak_reserved, dtype=np.int64),
			})

		return info