import collections
import typing

import torch

from s2p.planners.planner_base import PlannerBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.policy_model_base import PolicyModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.utils import gumbel_softmax_sample

class MPPIPlanner(PlannerBase):
	"""
	Class implementing MPPI planning. Based on TD-MPC2 MPPI implementation.
	"""
	def __init__(self, cfg, task:TaskBase):
		super().__init__(cfg=cfg, task=task)		
		# The bounds sampled actions are clamped to, taken from the task rather than assumed:
		# what a proposal has to respect is a property of the environment it will be executed
		# in, and only the task knows it. Kept at whatever shape the task reports so they
		# broadcast against the sampled `(horizon, num_samples, *action_dimension)`, and read
		# once here since a task's action space does not change. Infinities are a valid answer
		# -- for an env that enforces nothing, the clamp is meant to be a no-op.
		action_low, action_high = task.action_limits
		self.action_low = torch.as_tensor(action_low, dtype=torch.float, device=cfg.device)
		self.action_high = torch.as_tensor(action_high, dtype=torch.float, device=cfg.device)

		# Off unless the config asks for it; `PlannerConvergenceEvaluator` also turns it on for
		# the duration of its own plans. See `PlannerBase.collect_convergence_info`.
		if "collect_convergence_info" in cfg:
			self.collect_convergence_info = bool(cfg.collect_convergence_info)
		self.cfg = cfg
		self.task = task

		# Which of `plan`'s metrics elites are selected on, resolved once here rather than
		# per MPPI iteration -- and with `in` rather than a defaulted `getattr`, because on
		# an OmegaConf config a missing-key `getattr` is not free. OmegaConf raises
		# internally and swallows the error to return the default, and the caught exception's
		# traceback references the frame it was raised from: `plan`, whose locals hold the
		# flattened population. That is a reference cycle owning `num_envs * num_samples`
		# latents, which only the cycle collector can break -- and `TrainingRunner.run` calls
		# `gc.disable()`, collecting only every `log_interval` iterations, which its inner
		# collect/fetch loop never reaches mid-round. Measured upstream at 64 envs x 512
		# samples: 77 MiB retained per plan, versus 9 MiB total once the key is read this way.
		self.planning_criteria = (
			cfg.planning_criteria if "planning_criteria" in cfg else "total_value"
		)

	@staticmethod
	def _flatten_population(z, population_size):
		"""
		`[num_envs, D]` -> `[num_envs * population_size, D]`, env-major.

		Row `i * population_size + j` is env `i`'s latent, copied for its `j`-th trajectory. This
		is the *one* place the envs are flattened together, and env-major is a contract the rest
		of the planner reads back: `plan` un-flattens with `.reshape(num_envs, population_size)`.
		The sample-major alternative, `z.repeat(population_size, 1)`, produces the same *shape*
		with row `j * num_envs + i`, so swapping one for the other raises nothing and quietly
		scores every env against another env's trajectories -- which is why the flatten lives
		here rather than being written out at each of its two call sites.
		"""
		num_envs, latent_dim = z.shape
		return z.unsqueeze(1).expand(num_envs, population_size, latent_dim).reshape(
			num_envs * population_size, latent_dim
		)

	def _estimate_value(self, dynamics_model, reward_model, value_model, z, actions, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions.

		Rolls out a flat population of `M` trajectories; `plan` calls this with the `num_envs`
		envs' populations flattened together, `M = num_envs * num_samples`, so nothing here
		needs an env axis. Both arguments are already flat:
			z = [M, D]
			actions = [T, M, A]
		"""

		# Accumulators for summing over time. Sized off `z` rather than off `cfg.num_samples`,
		# which is one env's population and not the number of trajectories being rolled out.
		termination = torch.zeros_like(z[:, :1])
		G          = torch.zeros_like(z[:, :1])      # [N,1]
		R          = torch.zeros_like(z[:, :1])
		discount   = torch.ones(1, device=z.device)  # scalar tensor
		epi_dyn    = torch.zeros_like(z[:, :1])

		for t in range(self.cfg.horizon):
			# reward_ens = math.two_hot_inv(reward_ens, self.cfg) removed because the preference model is deterministic
			# Reward prediction: reward =  [N , 1] , reward_epi_uncer = [N , 1] , reward_aleatoric_uncer = [N, 1]
			extrinsic_reward = reward_model.reward(z.unsqueeze(0), actions[t].unsqueeze(0)).squeeze(0)
			# Dynamics prediction
			# Next State prediction: reward =  [N , D] , reward_epi_uncer = [N , 1] , reward_aleatoric_uncer = [N, 1]
			z = dynamics_model.dynamics(z.unsqueeze(0), actions[t].unsqueeze(0)).squeeze(0)

			# Adjusted reward (reward bonus shaping)
			adjusted_reward = extrinsic_reward
			G = G + discount * (1 - termination) * adjusted_reward

			# Discount update
			discount = discount * self.cfg.discount

			# Sum all quantities over time [N,1]
			R = R + extrinsic_reward
		# Bootstrap value from final state
		if self.cfg.use_value:
			# TODO: Come up with a better way to know if a model has a value function
			if value_model is not None:
				if isinstance(value_model, StateValueModelBase):
					Q = value_model.state_value(z.unsqueeze(0)).squeeze(0)
					value = G + discount * (1 - termination) * Q
				elif isinstance(value_model, StateActionValueModelBase):
					raise NotImplementedError("StateActionValueModel not yet supported for MPPI.")
			else:
				raise ValueError("`use_value=true` but value_model is None.")
		else:
			value = R
		# Info for logging
		info = {
			"total_return" : G,
		}
		value = value.nan_to_num(0)
		return value, info


	def plan(
			self, 
			dynamics_model:DynamicsModelBase, 
			reward_model:RewardModelBase, 
			value_model:typing.Union[StateValueModelBase, StateActionValueModelBase], 
			policy_model:PolicyModelBase,
			current_state:torch.Tensor, 
			eval_mode:bool, 
			action_prior:torch.Tensor) -> typing.Tuple[torch.Tensor, typing.Dict]:
		# Strip away the time dimension, leaving one latent per env: [E, D]
		z = current_state[0]
		num_envs = z.shape[0]

		# The batch structure of the answer follows the batch structure of the prior: a
		# `[horizon, action_dim]` prior gets a `[horizon, action_dim]` plan back, a
		# `[horizon, num_envs, action_dim]` one gets a plan per env. The body below is always
		# batched; an un-batched caller is one env wearing a size-1 axis, added here and taken
		# off on the way out. `num_envs` comes from the state rather than from the prior, so a
		# caller planning for several envs with one env's prior is told so instead of having its
		# prior silently broadcast across envs.
		batched = action_prior.ndim == 3
		if not batched:
			if num_envs != 1:
				raise ValueError(
					f"`current_state` describes {num_envs} envs but `action_prior` has shape "
					f"{tuple(action_prior.shape)}, which is one env's prior. Pass a "
					f"[horizon, {num_envs}, *action_dimension] prior to plan for all of them."
				)
			action_prior = action_prior.unsqueeze(1)

		horizon = self.cfg.horizon
		num_samples = self.cfg.num_samples
		num_elites = self.cfg.num_elites
		action_dimension = tuple(self.task.action_dimension)
		# Row selector for the paired advanced indexing below: `x[env_index[:, None], idx]` with
		# `idx` of shape [E, K] takes elite `k` *of env i* from env `i`'s row. Plain `x[:, idx]`
		# would instead take every env's row at every elite index, silently mixing the envs.
		env_index = torch.arange(num_envs, device=self.cfg.device)

		# Get the seed actions from the policy
		num_pi_trajs = int(self.cfg.fraction_of_policy_trajectories * num_samples)
		if not isinstance(policy_model, PolicyModelBase):
			# Nothing to seed from, so the whole population comes from the sampling
			# distribution. Zeroed rather than left at the configured fraction because the
			# fraction only describes how the population is split once there is a policy.
			num_pi_trajs = 0
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_envs, num_pi_trajs, *action_dimension, device=self.cfg.device)
			_z = self._flatten_population(z, num_pi_trajs)
			for t in range(horizon-1):
				action = policy_model.policy(_z)
				pi_actions[t] = action.view(num_envs, num_pi_trajs, *action_dimension)
				_z = dynamics_model.dynamics(_z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
			pi_actions[-1] = policy_model.policy(_z).view(num_envs, num_pi_trajs, *action_dimension)

		# Initialize state and parameters
		# Repeated State [E*N, D]
		z = self._flatten_population(z, num_samples)
		#Mean for action [T, E, A]
		mean = action_prior
		#std for action [T, E, A]
		std = torch.full((horizon, num_envs, *action_dimension), self.cfg.max_std, dtype=torch.float, device=self.cfg.device)

		#Actions are [T, E, N, A], the first `num_pi_trajs` of each env's N seeded by the policy
		actions = torch.empty(size=(horizon, num_envs, num_samples, *action_dimension), device=self.cfg.device)
		if num_pi_trajs > 0:
			actions[:, :, :num_pi_trajs] = pi_actions

		# One list per convergence metric, appended to per iteration and stacked at the end.
		# Left empty unless `collect_convergence_info`: the trace costs plan latency, which the
		# control loop pays on every step and `RealTaskEvaluator` turns into a control rate.
		trace = collections.defaultdict(list)

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions for non policy sampled actions (empty ones)
			r = torch.randn(size=(horizon, num_envs, num_samples - num_pi_trajs, *action_dimension), device=std.device)
			# r = torch.load(f"/path/to/other_project/baselines/evaluation/scripts/dstl/random.pt")
			# RBF-TAG
			actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
			actions_sample = actions_sample.clamp(self.action_low, self.action_high)
			actions[:, :, num_pi_trajs:] = actions_sample

			# Compute value, reward, and associated uncertainty info. The envs' populations are
			# rolled out as one flat batch of E*N: the models are batch-shape agnostic, and a
			# single call keeps the GPU busy where E separate ones would not.
			# value : [E*N,1] , infos [E*N, 1]
			value, info = self._estimate_value(
				dynamics_model,
				reward_model,
				value_model,
				z,
				actions.reshape(horizon, num_envs * num_samples, *action_dimension),
				eval_mode)

			# Define metrics with default fallback to "ubp_reward"
			planning_metric_map = {
				"total_value": value.reshape(num_envs, num_samples),
			}
			metric_values = planning_metric_map.get(self.planning_criteria, planning_metric_map["total_value"]) # [E, N]

			# Top-k selection, within each env
			elite_idxs = torch.topk(metric_values, num_elites, dim=1).indices  # [E, num_elites]

			# Extract elite values and actions
			elite_value = value.reshape(num_envs, num_samples, 1)[env_index[:, None], elite_idxs]   # [E, num_elites, 1]
			elite_actions = actions[:, env_index[:, None], elite_idxs]                              # [T, E, num_elites, A]

			# Extract elite info tensors. Reshaped to [E, N] first for the same reason the value
			# is: they come back flat, and indexing them with within-env elite indices without
			# that would read env 0's rows for every env.
			elite_info = {
				key: item.reshape(num_envs, num_samples, -1)[env_index[:, None], elite_idxs]
				for key, item in info.items()
			}                                                                                       # [E, num_elites, 1]

			# Elite scoring, per env: the softmax shift, the normalization and the weighted
			# moments below are all taken within an env, so no env's weights depend on another's.
			elite_metric = metric_values[env_index[:, None], elite_idxs].unsqueeze(-1)  # [E, num_elites, 1]
			max_metric = elite_metric.max(dim=1, keepdim=True).values                   # [E, 1, 1]
			score = torch.exp(self.cfg.temperature * (elite_metric - max_metric))       # [E, num_elites, 1]
			score = score / score.sum(dim=1, keepdim=True)
			previous_mean = mean
			normalizer = score.sum(dim=1) + 1e-9                                        # [E, 1]
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=2) / normalizer  #[1,E,K,1] * [T,E,K,A] -> [T,E,K,A], sum dim 2 -> [T,E,A]
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2) / normalizer).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std) 							#[T,E,A]

			# Record what this iteration did, for the convergence info dict
			if self.collect_convergence_info:
				# Value of the sampled population, and of the elites picked out of it
				trace["value_mean"].append(metric_values.mean(dim=1).mean())
				trace["value_std"].append(metric_values.std(dim=1).mean())
				trace["value_max"].append(metric_values.max(dim=1).values.mean())
				trace["elite_value_mean"].append(elite_metric.flatten(1).mean(dim=1).mean())
				trace["elite_value_std"].append(elite_metric.flatten(1).std(dim=1).mean())
				trace["elite_total_return_mean"].append(elite_info["total_return"].flatten(1).mean(dim=1).mean())
				# Fraction of the elites which came from the policy's seed trajectories
				# rather than from the sampling distribution: how much the plan owes to
				# its prior.
				trace["policy_elite_fraction"].append((elite_idxs < num_pi_trajs).float().mean())
				# Effective sample size of the softmax weights as a fraction of
				# `num_elites`: 1 when every elite contributes equally, 1/num_elites when
				# one dominates. A collapsed ESS means the next mean is really a single
				# trajectory.
				trace["elite_weight_ess_fraction"].append(
					(1.0 / (score.squeeze(-1).square().sum(dim=1) * num_elites)).mean()
				)
				# Width of the sampling distribution, which is what shrinking towards
				# `min_std` on its own looks like
				trace["action_std_mean"].append(std.mean())
				trace["action_std_max"].append(std.max())
				# RMS movement of the action mean over this iteration: the plan has settled
				# once this is small, whatever the value curve is doing.
				trace["action_mean_shift"].append((mean - previous_mean).square().mean().sqrt())
				# Profiles rather than scalars: the std of each planning step, and the mean
				# of the action which actually gets executed. Both are averaged over envs, so
				# they keep their shape ([horizon] and [action_dim]) at any `num_envs`.
				trace["action_std_per_step"].append(std.flatten(1).mean(-1))
				trace["first_action_mean"].append(mean[0].mean(dim=0))

		rand_idx = gumbel_softmax_sample(score.squeeze(-1), dim=1)                    # [E]
		sampled_actions = elite_actions[:, env_index, rand_idx]                       # [T, E, A]
		# Final info dict: the convergence trace stacked over iterations (empty unless
		# `collect_convergence_info`), plus the value of the plan which was actually returned
		info = {key: torch.stack(values) for key, values in trace.items()}
		info.update({
			"value":  elite_value[env_index, rand_idx], #value of the random action chosen from the elite actions, [E, 1]
			"total_return": elite_info["total_return"][env_index, rand_idx], #reward of the random action chosen from the elite actions, [E, 1]
		})
		# a, std= actions[0], std[0]

		# Different things can be done here:
		# TD-MPC2: 
		# 	`actions` is sampled from `elite_actions`, according to a Gumbel Softmax distribution based on the plan scores.
		#	In eval, just return `actions`; in train, return actions + std
		# My approach (based on traditional MPPI):
		#	In eval, return the mean of the final action distribution
		#	In train, return mean + std
		plan = mean
		# if eval_mode:
		# 	plan = mean + std * torch.randn_like(std, device=std.device)
		# else:
		# 	plan = sampled_actions + std * torch.randn_like(std, device=std.device)
		# 	# plan = mean + std * torch.randn_like(std, device=std.device)

		if batched:
			return plan, info
		# Drop the env axis again, from the plan and from the per-env info entries alike: a caller
		# that planned for one env with a one-env prior gets back the [horizon, *action_dimension]
		# plan it used to and can keep feeding it back in as its next prior, and the info dict is
		# as much a caller contract as the plan is -- `PlannerConvergenceEvaluator` silently skips
		# entries whose rank is not what it expects.
		info["value"] = info["value"].squeeze(1).squeeze(0)
		info["total_return"] = info["total_return"].squeeze(1).squeeze(0)
		return plan.squeeze(1), info
