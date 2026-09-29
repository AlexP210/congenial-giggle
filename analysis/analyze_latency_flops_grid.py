"""Combined stages 1-2 of the evaluation suite: FLOPs and wall-clock plan latency, swept
over a horizon x iterations x (num_samples, num_elites) grid for one model.

Driven by the same top-level Hydra configs as `analysis.py` (any `evaluate_*.yaml`), and
named by the same `runner.cfg.run_name`. The grid is not baked into this script: it comes
from four extra config values, all lists, of which the last two are paired by index
rather than crossed --

    +horizon_values=[1,2,3]
    +iterations_values=[1,2,3]
    +num_samples_values=[64,128]
    +num_elites_values=[8,16]

-- so index i of the last two means (num_samples=num_samples_values[i],
num_elites=num_elites_values[i]), matching how `submit_jobs_s2p_analysis.sh` pairs them
for stage 3.

The agent is loaded once and reused across every cell, since none of the four swept
values changes *which model* is being measured -- only `MPPIPlanner.plan`'s own tensors,
which it (re)allocates from `self.cfg` on every call (`self.cfg.num_samples` etc., read
fresh each time, not cached from construction). Mutating `model.planner.cfg` between
cells is exactly what a fresh `analyze_latency.py` invocation per cell would also end up
doing, just without paying container startup and checkpoint loading again for every one.
The per-module FLOPs counts (`encoder_flops` etc.) are likewise measured once: they are
properties of the model at batch size 1, not of the planner config, so nothing about the
grid changes them -- see `analyze_flops.py`'s docstring for what they mean and why the
population is not charged for.

Each cell's `flops`, `latency` and `memory` entries are appended to
`analysis/results/<run_name>.json` (see `results_store.py`) as soon as that cell finishes,
not held until the end: a kill partway through the grid -- Slurm's time limit, `scancel`
-- loses at most the one cell in flight, not every cell already measured.

Only the parts needed for FLOPs counting and timing are built: the agent (`cfg.agent`)
and the plan-latency evaluator (`cfg.evaluators.plan_latency`). The runner is never
instantiated, as in `analyze_latency.py` and `analyze_flops.py`.

No `lighting_preset` or `replan_every` handling here: neither affects FLOPs or latency
(see `submit_jobs_s2p_latency_flops.sh`), and this script never builds a task env for
rollouts, only for timing plans.

Example:

    python analysis/analyze_latency_flops_grid.py \
        --config-name=evaluate_s2p_push_cube \
        runner.cfg.run_name=Analyze_S2P_PushCube \
        hydra.run.dir=/path/to/outputs/hydra \
        data_dir=/path/to/datasets \
        checkpoint_dir=/path/to/pretrained_checkpoints \
        project_root=/path/to/project \
        output_dir=/path/to/outputs \
        device=cuda:0 \
        +horizon_values=[1,2,3] \
        +iterations_values=[1,2,3] \
        +num_samples_values=[64,128] \
        +num_elites_values=[8,16]
"""

import itertools
import os
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from s2p import PROJECT_ROOT
from s2p.evaluators.flops_evaluator import FLOPSEvaluator
from s2p.evaluators.plan_latency_evaluator import PlanLatencyEvaluator
from s2p.models.agent_model import AgentModel
from s2p.models.base.policy_model_base import PolicyModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase

import results_store

# Matches `s2p/main.py`, which is the path these models are actually run under; TF32
# matmuls change the latency being measured, so the setting has to be the one deployment
# uses rather than PyTorch's default.
MATMUL_PRECISION = "high"


def _sync(device):
	"""Drain the CUDA queue so work is charged to the config that issued it."""
	if torch.device(device).type == "cuda":
		torch.cuda.synchronize(device)


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	torch.set_float32_matmul_precision(MATMUL_PRECISION)

	# Seeding as `RunnerBase` does it: the runner is skipped here, but the reset states the
	# planner is timed from should still be reproducible.
	torch.manual_seed(cfg.seed)
	torch.cuda.manual_seed_all(cfg.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	run_name = cfg.runner.cfg.run_name

	horizon_values = [int(h) for h in cfg.horizon_values]
	iterations_values = [int(i) for i in cfg.iterations_values]
	num_samples_values = [int(n) for n in cfg.num_samples_values]
	num_elites_values = [int(n) for n in cfg.num_elites_values]
	if len(num_samples_values) != len(num_elites_values):
		raise ValueError(
			f"num_samples_values (len {len(num_samples_values)}) and num_elites_values "
			f"(len {len(num_elites_values)}) must be the same length -- they are paired "
			f"by index, not crossed"
		)
	samples_elites_pairs = list(zip(num_samples_values, num_elites_values))
	grid = list(itertools.product(horizon_values, iterations_values, samples_elites_pairs))
	if not grid:
		raise ValueError(
			"horizon_values, iterations_values, num_samples_values and num_elites_values "
			"must all be non-empty -- got "
			f"{len(horizon_values)}, {len(iterations_values)}, {len(num_samples_values)}, "
			f"{len(num_elites_values)} respectively"
		)

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	gpu = results_store.gpu_name(cfg.device)
	planner_cfg = model.planner.cfg

	# --- FLOPs: everything here is a property of the model at batch size 1, not of the
	# planner config, so it is measured once and reused arithmetically for every cell --
	# see `analyze_flops.py`, which this mirrors exactly (module docstring included).
	flops_evaluator = FLOPSEvaluator(
		cfg=OmegaConf.create({"evaluation_interval": None, "device": cfg.device})
	)
	module_info = flops_evaluator(model)

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

	def module_flops(name, present):
		return float(module_info[f"{name}/flops"]) if present else 0.0

	encoder_flops = module_flops("encoder", model.encoder_model is not None)
	dynamics_flops = module_flops("dynamics", model.dynamics_model is not None)
	reward_flops = module_flops("reward", model.reward_model is not None)
	value_flops = module_flops("value", model.value_model is not None)
	policy_flops = module_flops("policy", policy_is_planner_visible)

	use_value = planner_cfg.use_value
	fraction_of_policy_trajectories = planner_cfg.fraction_of_policy_trajectories

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

	# --- Latency: build the evaluator once, and pay CUDA context creation / lazy kernel
	# loads / allocator growth once too, before any cell is measured -- see
	# `analyze_latency.py`.
	plan_latency_evaluator: PlanLatencyEvaluator = instantiate(cfg.evaluators.plan_latency)
	_ = plan_latency_evaluator(model)
	_sync(cfg.device)

	# `+skip_existing=true` resumes an interrupted sweep: cells that already have all three
	# stages on record for this GPU are left alone, and a cell that was cut off after its
	# `flops` entry but before `latency`/`memory` is re-measured without a second `flops`.
	skip_existing = bool(cfg.get("skip_existing", False))
	existing = results_store.load(run_name) if skip_existing else {}

	def recorded(key, stage):
		entries = existing.get(key, {}).get(stage, [])
		if stage == "flops":
			return bool(entries)
		return any(entry.get("gpu") == gpu for entry in entries)

	print(f"Sweeping {len(grid)} configurations for {run_name} on {gpu}")

	for horizon, iterations, (num_samples, num_elites) in tqdm(grid, desc="Latency+FLOPs sweep"):
		planner_cfg.horizon = horizon
		planner_cfg.iterations = iterations
		planner_cfg.num_samples = num_samples
		planner_cfg.num_elites = num_elites

		key = results_store.planner_key(planner_cfg)
		if skip_existing and all(recorded(key, s) for s in ("flops", "latency", "memory")):
			continue

		# `MPPIPlanner.plan` zeroes the policy fraction when there is nothing to seed
		# from, whatever the config says; mirrored here so the two cannot disagree. The
		# only per-cell dependency: num_samples changes with the (num_samples,
		# num_elites) pair, so this is recomputed even though `fraction_of_policy_
		# trajectories` itself is a model constant.
		num_pi_trajs = (
			int(fraction_of_policy_trajectories * num_samples) if policy_is_planner_visible else 0
		)
		seeding = 0.0
		if num_pi_trajs > 0:
			seeding = horizon * policy_flops + (horizon - 1) * dynamics_flops
		rollout = horizon * (reward_flops + dynamics_flops)
		plan_flops = seeding + iterations * (rollout + terminal_flops_per_trajectory)
		total_flops = encoder_flops + plan_flops

		flops_path = results_store.results_path(run_name)
		if not (skip_existing and recorded(key, "flops")):
			results_store.add_entry(run_name, planner_cfg, "flops", {"flops": float(total_flops)})

		# Hand the previous cell's cached blocks back, so `peak_reserved` reflects this
		# config rather than the largest one swept so far.
		if torch.device(cfg.device).type == "cuda":
			torch.cuda.empty_cache()

		info = plan_latency_evaluator(model, verbose=False)
		_sync(cfg.device)
		latency_entry = {"gpu": gpu, "latencies": [float(t) for t in info["plan_latencies"]]}
		results_store.add_entry(run_name, planner_cfg, "latency", latency_entry)

		memory_msg = ""
		if "plan_memory_peak_allocated" in info:
			memory_entry = {
				"gpu": gpu,
				"baseline_allocated_bytes": int(info["plan_memory_baseline_allocated"]),
				"peak_allocated_bytes": [int(b) for b in info["plan_memory_peak_allocated"]],
				"peak_reserved_bytes": [int(b) for b in info["plan_memory_peak_reserved"]],
			}
			results_store.add_entry(run_name, planner_cfg, "memory", memory_entry)
			plan_bytes = max(memory_entry["peak_allocated_bytes"]) - memory_entry["baseline_allocated_bytes"]
			memory_msg = f", {plan_bytes / 2**20:.1f} MiB/plan"

		tqdm.write(
			f"{results_store.planner_key(planner_cfg)}: {total_flops / 1e9:.3f} GFLOPs, "
			f"{info['plan_latency'] * 1e3:.1f} ms{memory_msg} -- appended to {flops_path.name}"
		)

	print(f"swept {len(grid)} configurations for {run_name} -> {results_store.results_path(run_name)}")


if __name__ == "__main__":
	main()
