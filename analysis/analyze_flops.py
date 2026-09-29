"""Stage 1 of the evaluation suite: the critical-path FLOPs of a plan under the planner
configuration in the Hydra config -- no sweep, one configuration per invocation.

Driven by the same top-level Hydra configs as `analysis.py` (any `evaluate_*.yaml`), and
named by the same `runner.cfg.run_name`.

Nothing is timed here. Each of the agent's modules is counted once, at batch size 1, and
the cost of a plan is then assembled from the call counts `MPPIPlanner.plan` actually
makes for the configured horizon and iterations. That is what makes this stage cheap: it
is arithmetic over five measurements, not a measured plan.

The count is along the plan's *critical path*: what one trajectory costs, not what the
whole population costs. MPPI's `num_samples` trajectories are independent -- the planner
rolls them out as one batched call, and nothing couples one row to another -- so on
sufficiently wide hardware the population is a parallel axis rather than a longer
computation, and a hardware-independent cost should not charge for it. The two axes that
are genuinely sequential, horizon and iterations, are charged for: each MPPI iteration
needs the previous iteration's elites, and each rollout step needs the previous step's
latent. So, per plan, for one env and one trajectory:

    encode                                                      1  encoder call
    policy seeding (once, when the population is policy-seeded)
                                              horizon           policy calls
                                        (horizon-1)             dynamics calls
    per MPPI iteration:
                                              horizon           reward calls
                                              horizon           dynamics calls
                                                    1           value call   (use_value)
                                                    1           policy call  (use_value
                                                                  with a Q(s,a) head)

This is a deliberate choice, and it is a floor rather than a measurement: whether the
population is really free depends on the model and the device. Stage 2
(`analyze_latency.py`) is what measures the cost as actually paid; this is the work that
cannot be parallelised away.

`num_pi_trajs` is `int(fraction_of_policy_trajectories * num_samples)`, and zero when the
agent has no policy to seed from -- both exactly as the planner computes them. It decides
*whether* the seeding term applies, but does not scale it, for the same reason
`num_samples` does not.

Only the agent (`cfg.agent`, which carries the planner and loads its checkpoint) is built.
The runner is never instantiated -- it would also build the task evaluators, and with them
an env and the offline dataset -- but `cfg.runner.cfg.run_name` still names the output, so
the same `run_name=` override used elsewhere identifies the run here. The FLOPs evaluator
is constructed directly rather than taken from `cfg.evaluators.flops`: it holds no state
and needs no task, and most `evaluate_*.yaml` do not compose that group, so reading it from
the config would restrict this script to the handful that do. (`RealTaskEvaluator` builds
its own sub-evaluators the same way, for the same reason.)

The total FLOPs of a plan -- the planning search plus the one encoder call, matching what
`analyze_latency.py` times -- is appended to this run's entry in
`analysis/results/<run_name>.json` under the planner's own
`(num_samples, num_elites, horizon, iterations)` -- see `results_store.py`.

Example:

    python analysis/analyze_flops.py \
        --config-name=evaluate_s2p_push_cube \
        runner.cfg.run_name=Analyze_S2P_PushCube \
        hydra.run.dir=/path/to/outputs/hydra \
        data_dir=/path/to/datasets \
        checkpoint_dir=/path/to/pretrained_checkpoints \
        project_root=/path/to/project \
        output_dir=/path/to/outputs \
        device=cuda:0

The planner configuration is whatever `cfg.agent.planner` says -- override
`agent.planner.cfg.horizon=...` etc. from the command line to test a different one.
"""

import os
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from s2p import PROJECT_ROOT
from s2p.evaluators.flops_evaluator import FLOPSEvaluator
from s2p.models.agent_model import AgentModel
from s2p.models.base.policy_model_base import PolicyModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase

import results_store


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	# Seeding as `RunnerBase` does it. FLOP counts do not depend on it, but a model whose
	# construction draws (a fresh head, a random projection) should still be the one the
	# other two stages measure.
	torch.manual_seed(cfg.seed)
	torch.cuda.manual_seed_all(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	run_name = cfg.runner.cfg.run_name

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	flops_evaluator = FLOPSEvaluator(
		cfg=OmegaConf.create({"evaluation_interval": None, "device": cfg.device})
	)

	# Counts for the four modules the evaluator knows about. Its `inference_time` entries
	# are deliberately dropped: they are one un-warmed call each, and `analyze_latency.py`
	# is the stage that measures time.
	module_info = flops_evaluator(model)

	# The policy is counted here rather than by the evaluator, which has no policy branch.
	# It is part of a plan's cost twice over -- seeding the population, and supplying the
	# action a Q(s,a) head bootstraps from -- so leaving it out would under-count both.
	policy_is_planner_visible = isinstance(model.policy_model, PolicyModelBase)
	if policy_is_planner_visible:
		policy_input_dims, policy_function = model.policy_model.get_policy_function()
		policy_flops, policy_macs, policy_params, _ = flops_evaluator.benchmark(
			policy_function, policy_input_dims, cfg.device
		)
		module_info.update({
			"policy/flops": policy_flops,
			"policy/macs": policy_macs,
			"policy/params": policy_params,
		})

	# A module the agent does not have costs nothing, rather than being missing from the
	# arithmetic. Structural checks against the model, not defaults for absent keys: an
	# agent that has a module but whose count did not land is a bug, and should raise.
	def module_flops(name, present):
		return float(module_info[f"{name}/flops"]) if present else 0.0

	encoder_flops = module_flops("encoder", model.encoder_model is not None)
	dynamics_flops = module_flops("dynamics", model.dynamics_model is not None)
	reward_flops = module_flops("reward", model.reward_model is not None)
	value_flops = module_flops("value", model.value_model is not None)
	policy_flops = module_flops("policy", policy_is_planner_visible)

	# The planner configuration being characterised, read off the planner the agent was
	# built with -- no sweep, this is the one configuration `cfg.agent.planner` specifies.
	planner_cfg = model.planner.cfg
	horizon = planner_cfg.horizon
	iterations = planner_cfg.iterations
	num_samples = planner_cfg.num_samples
	use_value = planner_cfg.use_value
	# `MPPIPlanner.plan` zeroes the policy fraction when there is nothing to seed from,
	# whatever the config says; mirrored here so the two cannot disagree.
	num_pi_trajs = (
		int(planner_cfg.fraction_of_policy_trajectories * num_samples)
		if policy_is_planner_visible
		else 0
	)

	# A Q(s,a) head needs an action to score, which costs one policy call per trajectory on
	# top of the value call itself; a V(s) head does not.
	terminal_flops_per_trajectory = 0.0
	if use_value:
		if isinstance(model.value_model, StateValueModelBase):
			terminal_flops_per_trajectory = value_flops
		elif isinstance(model.value_model, StateActionValueModelBase) and policy_is_planner_visible:
			terminal_flops_per_trajectory = value_flops + policy_flops
		else:
			raise ValueError(
				"planner has `use_value: true`, but the agent's value model is neither a "
				"StateValueModelBase nor a StateActionValueModelBase paired with a policy -- "
				"`MPPIPlanner._estimate_value` would raise on the first plan."
			)

	# Seeding happens once per plan, before the MPPI loop, and the last step of the seed
	# rollout needs no dynamics call -- the action it produces is the last one.
	# `num_pi_trajs` decides whether the term applies at all (there is no seed rollout
	# without it) but does not scale it: those trajectories go through the planner as one
	# batched call, the same parallel axis as the population.
	seeding = 0.0
	if num_pi_trajs > 0:
		seeding = horizon * policy_flops + (horizon - 1) * dynamics_flops

	# One trajectory's rollout and terminal bootstrap. The `num_samples` of them are
	# independent and issued as a single batched call, so the critical path is one.
	rollout = horizon * (reward_flops + dynamics_flops)
	terminal = terminal_flops_per_trajectory

	plan_flops = seeding + iterations * (rollout + terminal)

	# What the plan-latency evaluator times is an encode followed by a plan, so the total
	# this stage reports against is the same thing.
	total_flops = encoder_flops + plan_flops

	entry = {"flops": float(total_flops)}
	output_path = results_store.add_entry(run_name, planner_cfg, "flops", entry)
	print(
		f"{run_name} {results_store.planner_key(planner_cfg)}: {total_flops / 1e9:.3f} "
		f"GFLOPs (encode {encoder_flops / 1e9:.3f}), population {num_samples} "
		f"({num_pi_trajs} policy-seeded, not charged -- this is a critical-path count), "
		f"use_value={use_value}, appended to {output_path}"
	)


if __name__ == "__main__":
	main()
