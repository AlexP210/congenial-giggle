import abc
import math
import typing
from collections import deque
import time

import torch
from collections import deque
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from omegaconf import OmegaConf

from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.models.agent_model import AgentModel
from s2p.evaluators.plan_latency_evaluator import PlanLatencyEvaluator
from s2p.evaluators.policy_latency_evaluator import PolicyLatencyEvaluator
from s2p.lib.utils import episode_boundary

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from tdmpc2.envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class RealTaskEvaluator(OnlineEvaluatorBase):
	def __init__(self, cfg:OmegaConf, task:OnlineTaskBase):

		super().__init__(cfg, task)
		self.cfg = cfg

		self.task = task

		# Episodes run `num_envs` at a time. Each env is an independent episode, so this only
		# changes how long the evaluation takes, not what it measures -- in particular not the
		# plan budget, which the latency evaluators below deliberately measure at num_envs=1.
		self.num_envs = int(self.cfg.num_envs)
		# `cfg.max_episode_steps` is this evaluator's own time limit, in the task's units
		# (primitive steps on ManiSkill); null keeps the task's registered one.
		self.env = task.make_env(
			num_envs=self.num_envs, max_episode_steps=self.cfg.max_episode_steps
		)
		self.best_return = float('-inf')
		self.best_success_once_rate = float('-inf')
		self.best_success_at_end_rate = float('-inf')

		# We need a latency evaluator to test the real-time performance
		self.plan_latency_evaluator = PlanLatencyEvaluator(
			cfg=OmegaConf.create(
				{
					"num_plans": 100,
					"device": self.cfg.device,
					"observation_key": None
				}
			),
			task=self.task
		)
		self.policy_latency_evaluator = PolicyLatencyEvaluator(
			cfg=OmegaConf.create(
				{
					"num_queries": 100,
					"device": self.cfg.device
				}
			),
			task=self.task
		)
		# Batched like every action the env takes: `action_dimension` is per-env, and the env
		# wears the batch axis every task now carries, whether it holds one episode or many.
		self.zero_action = torch.zeros((self.num_envs, *self.task.action_dimension), device=self.cfg.device)

	def _truncate(self, queue, n):
		"""Keep the leftmost `n` entries, in place. A deque cannot be sliced, and rebinding it
		to a list would lose both `maxlen` and `popleft`."""
		while len(queue) > n:
			queue.pop()          # from the right: the leftmost entries are the imminent ones

	def __call__(self, model:AgentModel) -> typing.Dict[str, typing.Any]:

		# Container for logging info we will output
		info = {}

		# Containers for per-episode statistics across episodes.
		# `success_once` is the fraction of episodes in which the success criterion fired
		# at any point; `success_at_end` is the fraction still satisfying it on the final
		# step. These differ because ManiSkill recomputes success from the current state
		# every step, so it does not latch — the cube can be pushed into the goal region
		# and back out again within one episode.
		returns = []
		successes_once = []
		successes_at_end = []

		# Env steps from episode start to the first step satisfying the success criterion. An
		# episode that never succeeds contributes its full length, so the mean stays finite and
		# a policy that fails is scored as "took the whole episode". Steps rather than seconds:
		# the control interval converts them whenever a wall-clock number is wanted.
		times_to_success = []

		# Tasks without a success criterion (e.g. dm_control) report no "success" key, in
		# which case the success metrics are omitted rather than reported as zero
		task_reports_success = False

		# Evaluation loop
		# A given `steps_needed_to_plan` (e.g. from latencies recorded ahead of time) is used
		# as-is, so the latency is only measured here when it has to be computed.
		if self.cfg.steps_needed_to_plan is not None:
			steps_needed_to_plan = self.cfg.steps_needed_to_plan
		else:
			if self.cfg.action_mode == "plan":
				time_to_plan = self.plan_latency_evaluator(model, verbose=False)["plan_latency"]
			elif self.cfg.action_mode == "act":
				time_to_plan = self.policy_latency_evaluator(model, verbose=False)["policy_latency"]
			control_interval = self.task.get_control_interval()
			steps_needed_to_plan = int(np.ceil(time_to_plan/control_interval))			

		# Episodes run `num_envs` at a time: the envs are independent, so a batch is just
		# `num_envs` of the episodes this used to run one after another. The last batch is
		# trimmed when `num_envs` does not divide `num_episodes`, so exactly `num_episodes` of
		# them are reported on rather than a number rounded up to the batch.
		num_envs = self.num_envs
		num_batches = math.ceil(self.cfg.num_episodes / num_envs)
		remaining = int(self.cfg.num_episodes)

		for batch_index in tqdm(range(num_batches), "Real Task Evaluation"):

			counted = min(num_envs, remaining)
			remaining -= counted

			# Initialize the first transition
			obs, done = self.env.reset(), False
			# Per-env accumulators, one entry per episode in this batch
			return_ = torch.zeros(num_envs, device=self.cfg.device)
			success_once = torch.zeros(num_envs, dtype=torch.bool, device=self.cfg.device)
			success_at_end = torch.zeros(num_envs, dtype=torch.bool, device=self.cfg.device)
			steps_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=self.cfg.device)
			steps = 0

			# (T=1, B, ...): the observation already carries the env axis, so only time is added.
			obs = obs.to(self.cfg.device).unsqueeze(0)

			# Get the first state estimate
			state = model.encoder_model.encode(obs)

			# Initialize video frames for the log
			if self.cfg.save_video:
				frames = [np.transpose(self.env.render(), (2, 0, 1))]

			# Initialize the action queue
			action_queue = deque(maxlen=steps_needed_to_plan+model.planner.cfg.horizon)

			# Initialize the first plan
			previous_plan = torch.zeros(size=(model.planner.cfg.horizon, num_envs, *self.task.action_dimension), device=self.cfg.device)

			# We will replan every `replan_counter` env steps, but since we are
			# modelling a real env, this can't be lower than `steps_needed_to_plan`
			replan_counter = 0

			# Step the environment
			while not done:

				# If it's time to generate a new plan
				if replan_counter == 0:

					replan_counter = steps_needed_to_plan + self.cfg.replan_every

					# Since we are modelling a real-time env, planning is not 
					# free - how are we going to fill `steps_needed_to_plan`?
					if self.cfg.interim_behaviour == "wait":
						action_queue.clear()
						action_queue.extend([self.zero_action,]*steps_needed_to_plan)
					elif self.cfg.interim_behaviour == "finish_plan_then_wait":
						self._truncate(action_queue, steps_needed_to_plan)
						action_queue.extend([self.zero_action,] * (steps_needed_to_plan - len(action_queue)))
					elif self.cfg.interim_behaviour == "finish_plan_then_repeat":
						self._truncate(action_queue, steps_needed_to_plan)
						last_action = action_queue[-1] if action_queue else self.zero_action
						action_queue.extend([last_action,] * (steps_needed_to_plan - len(action_queue)))

					# Generate the plan
					if self.cfg.action_mode == "plan":
						action_prior = torch.zeros_like(previous_plan)
						action_prior[:-self.cfg.replan_every] = previous_plan[self.cfg.replan_every:]
						plan, _ = model.plan(state=state, action_prior=action_prior)
						previous_plan = plan
						action_queue.extend(list(plan.unbind(0)))
					else:
						actions = model.act(state=state)
						action_queue.append(actions[0])

				# Get the next action
				action = action_queue.popleft()	
				replan_counter -= 1

				# Step the environment
				obs, reward, terminated, truncated, step_info = self.env.step(action.cpu().detach())
				obs = obs.to(self.cfg.device).unsqueeze(0)
				done = episode_boundary(terminated, truncated)
				steps += 1

				# Accumulate episode returns, per env
				return_ += reward.to(self.cfg.device).reshape(num_envs)

				# Track success both as "ever satisfied" and "satisfied on the last step".
				# `in` is used rather than `.get`, since dm_control tasks return a
				# defaultdict whose missing keys would otherwise materialise as zeros.
				if "success" in step_info:
					task_reports_success = True
					success_at_end = torch.as_tensor(step_info["success"], device=self.cfg.device).reshape(num_envs).bool()
					success_once = success_once | success_at_end
					# First step at which each env succeeded; -1 until it does.
					first = success_at_end & (steps_to_success < 0)
					steps_to_success = torch.where(first, torch.full_like(steps_to_success, steps), steps_to_success)

				if self.cfg.save_video:
					frames.append(np.transpose(self.env.render(), (2, 0, 1)))

				# Update the state estimate
				action = action.unsqueeze(0)
				state = model.encoder_model.encode(obs, state, action)

			# An episode that never succeeded contributes its full length, so the mean stays
			# finite and a failure is scored as "took the whole episode".
			times = torch.where(steps_to_success < 0, torch.full_like(steps_to_success, steps), steps_to_success)

			# `[:counted]` drops the envs the trimmed final batch ran but does not report on.
			returns.extend(return_[:counted].tolist())
			successes_once.extend(success_once[:counted].tolist())
			successes_at_end.extend(success_at_end[:counted].tolist())
			times_to_success.extend(times[:counted].tolist())

		mean_return = float(np.mean(returns))
		if mean_return > self.best_return:
			self.best_return = mean_return
			self._save_best_checkpoint(model, "best_return")

		info.update({
			f"steps_needed_to_plan": steps_needed_to_plan,
			f"episode_return": mean_return,
			f"episode_return_sem": np.std(returns)/np.sqrt(len(returns)),
			f"episode_return_distribution": np.array(returns, dtype=np.float32),
		})

		if task_reports_success:
			mean_success_once = float(np.mean(successes_once))
			mean_success_at_end = float(np.mean(successes_at_end))
			mean_time_to_success = float(np.mean(times_to_success))

			if mean_success_once > self.best_success_once_rate:
				self.best_success_once_rate = mean_success_once
				self._save_best_checkpoint(model, "best_success_once_rate")
			if mean_success_at_end > self.best_success_at_end_rate:
				self.best_success_at_end_rate = mean_success_at_end
				self._save_best_checkpoint(model, "best_success_at_end_rate")

			info.update({
				f"episode_success_once_rate": mean_success_once,
				f"episode_success_once_rate_sem": self._bernoulli_sem(successes_once),
				f"episode_success_once_rate_distribution": np.array(successes_once, dtype=np.float32),
				f"episode_success_at_end_rate": mean_success_at_end,
				f"episode_success_at_end_rate_sem": self._bernoulli_sem(successes_at_end),
				f"episode_success_at_end_rate_distribution": np.array(successes_at_end, dtype=np.float32),
				f"episode_time_to_success": mean_time_to_success,
				f"episode_time_to_success_sem": np.std(times_to_success)/np.sqrt(len(times_to_success)),
				f"episode_time_to_success_distribution": np.array(times_to_success, dtype=np.float32),
			})

		if self.cfg.save_video:
			info.update({
				f"video": np.array(frames)
			})

		return info