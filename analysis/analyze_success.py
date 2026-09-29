"""Stage 3 of the evaluation suite: success rate under the real task, for the planner
configuration in the Hydra config -- no sweep, one configuration per invocation.

Driven by the same top-level Hydra configs as `analysis.py` (any `evaluate_*.yaml`), and
named by the same `runner.cfg.run_name`.

This is the expensive stage, and the only one that has to run rollouts. It uses
`RealTaskEvaluator`, which models planning as costing real time: for `steps_needed_to_plan`
env steps after a replan is issued the controller has no new action to execute, and fills
those steps according to `interim_behaviour`. That step count is the reason this stage
depends on stage 2 -- it is the plan latency divided by the task's control interval -- and
it is where a slow planner is actually penalised, so the whole result rests on which
latency numbers go in.

Rather than let the evaluator measure latency again itself -- inside a rollout, on a
machine that is by then also running a simulator -- the latencies `analyze_latency.py`
already recorded for this exact `(num_samples, num_elites, horizon, iterations)`, on
`latency_gpu`, are read back from `analysis/results/<latency_run_name>.json` and converted
to steps here, and handed to the evaluator as `steps_needed_to_plan`. Scoped to a single
GPU -- and not, say, averaged across whichever GPUs happened to record one -- because
latency is hardware-dependent: a budget built by averaging across devices would describe
none of them. `latency_gpu` defaults to whatever GPU this run is actually executing on
(`cfg.device`), but can be pointed at a different recorded GPU with `+latency_gpu=<name>`
when it does not matter whether the plan budget reflects the hardware the rollout itself
runs on.

`latency_run_name` defaults to this run's own `run_name`, i.e. stage 2 and this stage
share one results file, as in the example below. Pass `+latency_run_name=<name>` to read
from a different one instead -- for a sweep that varies something latency does not
actually depend on (e.g. the task's `lighting_preset`), where stage 2 was run once under
a name that leaves that out, and every variant's stage 3 should draw from that same
recording rather than each re-measuring an identical latency.

Only the parts needed for the rollouts are built: the agent (`cfg.agent`, which carries the
planner and loads its checkpoint) and the real-task evaluator
(`cfg.evaluators.real_task_planning`). The runner is never instantiated -- it would also
build the remaining evaluators and the offline dataset -- but `cfg.runner.cfg.run_name` is
still what names the output entry.

The number of episodes run, the `replan_every` they were run with, the GPU whose recorded
latencies sized the plan budget, and per episode whether it ever satisfied the success
criterion, whether it was still satisfied on the final step, how many env steps success
took, and the episode return, are appended to this run's entry in
`analysis/results/<run_name>.json` under the planner's own `(num_samples, num_elites,
horizon, iterations)` -- see `results_store.py`. `replan_every` and `gpu` are recorded on
the entry rather than folded into that key: `replan_every` is
`evaluators.real_task_planning.cfg.replan_every`, not a planner setting, and unlike the
other four it changes only what a rollout does with a plan, not the plan itself; `gpu` is
`latency_gpu` -- which of possibly several recorded latency sources this particular run
drew its plan budget from, not necessarily the hardware the rollout itself ran on. Neither
has any bearing on `analyze_latency.py` or `analyze_flops.py`, so neither belongs in a key
those two stages also write under.

Success is recorded as "ever satisfied" rather than "still satisfied on the final step",
since ManiSkill recomputes success from the current state every step rather than latching
it -- an episode can satisfy the criterion and then stop. An episode that never succeeds
contributes its full length to `time_to_success`.

Example:

    python analysis/analyze_success.py \
        --config-name=evaluate_s2p_push_cube \
        runner.cfg.run_name=Analyze_S2P_PushCube \
        hydra.run.dir=/path/to/outputs/hydra \
        data_dir=/path/to/datasets \
        checkpoint_dir=/path/to/pretrained_checkpoints \
        project_root=/path/to/project \
        output_dir=/path/to/outputs \
        device=cuda:0

`num_episodes` is whatever the evaluator config says (50 by default) and is not overridden
here, so it can be lowered from the command line with
`evaluators.real_task_planning.cfg.num_episodes=...`. The planner configuration is
whatever `cfg.agent.planner` says -- it must match a configuration `analyze_latency.py`
has already recorded for this `run_name`, or this raises rather than measuring latency
itself; see the module docstring above.

`+feasible_configs=<json>` scores every `feasible` row of a `feasible_planner_configs.py`
report instead -- its planner fields and `replan_every` -- in this one process, reusing the
model and env rather than building both per config. Each config is reseeded and starts from
the same episodes, and each is appended to the results file as it finishes. With
`+skip_existing=true`, configs that already have a success entry for that `replan_every` on
`latency_gpu` are skipped, so a sweep cut off by a time limit resumes where it stopped.

`+accumulate=true` instead adds episodes to what is already on record, so repeated
submissions shrink the error bars. A config's rounds are its distinct seeds on record for
that `replan_every` on `latency_gpu`. Each invocation brings every config up to one round
more than the least-sampled config has, and skips any config already there. So a plain
resubmission always does the useful thing: after a complete sweep it starts a new round,
and after one a time limit cut off it finishes the round that job left. A new round runs
under the smallest seed at or above `cfg.seed` not yet on record for that config, so no
episodes are ever repeated, and configs in the same round share a seed and so the same
episodes, as in a single sweep. The seed a round ran under is recorded in
`settings.seed`. Supersedes `+skip_existing`.
"""

import json
import os
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from s2p import PROJECT_ROOT
from s2p.evaluators.real_task_evaluator import RealTaskEvaluator
from s2p.models.agent_model import AgentModel

import results_store

# Matches `s2p/main.py`, which is the path these models are actually run under, so this
# stage plans as fast as it can inside a rollout under deployment's numerics.
MATMUL_PRECISION = "high"


def to_jsonable(value):
	"""`value` with arrays, tensors and numpy scalars turned into plain lists and numbers."""
	if isinstance(value, dict):
		return {str(k): to_jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [to_jsonable(v) for v in value]
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().tolist()
	if isinstance(value, (np.ndarray, np.generic)):
		return value.tolist()
	return value


def seed_everything(seed):
	# Seeding as `RunnerBase` does it: the runner is skipped here, but the episodes the
	# model is scored on should still be reproducible.
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)


def sweep_configs(cfg, planner_cfg, task_evaluator):
	"""The (planner fields + `replan_every`) configurations to score: every `feasible` row of
	`+feasible_configs=<json>` (written by `feasible_planner_configs.py`), or else just the
	one the Hydra config describes."""
	if "feasible_configs" not in cfg:
		return [{
			**results_store.planner_dict(planner_cfg),
			"replan_every": int(task_evaluator.cfg.replan_every),
		}]
	with open(cfg.feasible_configs) as f:
		report = json.load(f)
	fields = (*results_store.PLANNER_FIELDS, "replan_every")
	return [
		{**{field: int(row[field]) for field in fields},
		 "feasible_steps_needed_to_plan": row.get("steps_needed_to_plan")}
		for row in report["feasible"]
	]


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	torch.set_float32_matmul_precision(MATMUL_PRECISION)
	seed_everything(cfg.seed)

	run_name = cfg.runner.cfg.run_name

	# `analyze_latency.py` may have been run under a different run_name than this success
	# run: latency and FLOPs do not depend on the task's `lighting_preset`, so a caller
	# sweeping lighting only for success can record latency once under a preset-agnostic
	# run_name and point every lighting variant's success run at that same latency data
	# with `+latency_run_name=`, rather than re-measuring identical latency per preset.
	# Defaults to this run's own name, which is exactly stage 2's behaviour when the two
	# scripts share one run_name.
	latency_run_name = cfg.latency_run_name if "latency_run_name" in cfg else run_name

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	planner_cfg = model.planner.cfg
	gpu = results_store.gpu_name(cfg.device)

	# The GPU whose recorded latencies size the plan budget. Defaults to the GPU this
	# rollout is actually running on, but can be pointed at a different recorded GPU with
	# `+latency_gpu=<name>` -- see the module docstring -- when the plan budget need not
	# reflect the hardware the rollout itself runs on.
	latency_gpu = cfg.latency_gpu if "latency_gpu" in cfg else gpu

	task_evaluator: RealTaskEvaluator = instantiate(cfg.evaluators.real_task_planning)
	num_episodes = task_evaluator.cfg.num_episodes

	# Rendering every frame of every episode costs real time, and this stage is already the
	# expensive one; the videos would go nowhere, since nothing here logs to wandb.
	task_evaluator.cfg.save_video = False

	# Seconds per env step, which is what turns a plan latency into a number of steps the
	# controller spends planning.
	control_interval = task_evaluator.task.get_control_interval()

	# `+feasible_configs=<json>` scores every feasible config in one process, against the
	# one model and env already built, rather than paying for both again per config.
	# `+skip_existing=true` resumes a sweep a time limit cut off: a config that already has
	# a success entry for this `replan_every` on `latency_gpu` is left alone.
	configs = sweep_configs(cfg, planner_cfg, task_evaluator)
	sweeping = "feasible_configs" in cfg
	accumulate = bool(cfg.get("accumulate", False))
	skip_existing = bool(cfg.get("skip_existing", False)) and not accumulate
	existing = results_store.load(run_name) if (skip_existing or accumulate) else {}

	def recorded_seeds(config):
		"""Seeds of this config's success entries for its `replan_every` on `latency_gpu`;
		None for an entry that does not say which seed it ran under."""
		key = results_store.planner_key(OmegaConf.create(
			{field: config[field] for field in results_store.PLANNER_FIELDS}))
		return {
			e.get("settings", {}).get("seed")
			for e in existing.get(key, {}).get("success", [])
			if e.get("replan_every") == config["replan_every"] and e.get("gpu") == latency_gpu
		}

	# `+accumulate=true`: bring every config up to one round past the least-sampled one --
	# see the module docstring.
	if accumulate:
		target_rounds = min(len(recorded_seeds(config)) for config in configs) + 1
		print(f"Accumulating: bringing every configuration up to {target_rounds} round(s)")

	if sweeping:
		print(f"Sweeping {len(configs)} configurations from {cfg.feasible_configs}")

	for index, config in enumerate(configs):
		for field in results_store.PLANNER_FIELDS:
			setattr(planner_cfg, field, config[field])
		task_evaluator.cfg.replan_every = config["replan_every"]
		key = results_store.planner_key(planner_cfg)
		seeds_on_record = recorded_seeds(config)
		if (skip_existing and seeds_on_record) or (accumulate and len(seeds_on_record) >= target_rounds):
			print(f"[{index + 1}/{len(configs)}] {key}, replan_every={config['replan_every']}: "
			      f"already on record ({len(seeds_on_record)} round(s)), skipped")
			continue

		# A fresh seed per round when accumulating, so a new round never replays episodes
		# already on record.
		seed = int(cfg.seed)
		if accumulate:
			while seed in seeds_on_record:
				seed += 1

		# Every config in a sweep starts from the same RNG state and the same episodes, as
		# it would in a process of its own: the global RNGs are reseeded, and a seeded reset
		# restarts ManiSkill's episode RNG, which the evaluator's unseeded resets then draw
		# from.
		if sweeping or accumulate:
			seed_everything(seed)
			task_evaluator.env.reset(seed=seed)

		# The plan budget for this configuration, from stage 2's recorded measurements on
		# `latency_gpu` rather than a fresh one taken while the simulator is running.
		# Averaged across every matching entry on file, and rounded up, as the evaluator
		# itself would: a plan that overruns a step boundary costs the whole next step.
		latencies = results_store.latencies_for(latency_run_name, planner_cfg, latency_gpu)
		latency = float(np.mean(latencies))
		steps = int(np.ceil(latency / control_interval))
		task_evaluator.cfg.steps_needed_to_plan = steps

		progress = f"[{index + 1}/{len(configs)}] " if sweeping else ""
		print(
			f"{progress}Scoring {key}, replan_every={config['replan_every']} ({num_episodes} "
			f"episodes, seed {seed}) for {run_name} on {gpu}, control interval {control_interval * 1e3:.1f} ms, "
			f"plan budget {latency * 1e3:.1f} ms ({steps} steps, from {len(latencies)} "
			f"recorded latencies on {latency_gpu!r} under run_name={latency_run_name})"
		)
		# feasible_planner_configs.py sizes its budget from the median (by default), this
		# from the mean, so the two can round to different step counts near a boundary.
		feasible_steps = config.get("feasible_steps_needed_to_plan")
		if feasible_steps is not None and feasible_steps != steps:
			print(f"note: {cfg.feasible_configs} has steps_needed_to_plan={feasible_steps} "
			      f"for this config; using {steps} from the mean recorded latency")

		info = task_evaluator(model)

		if "episode_success_once_rate_distribution" not in info:
			raise ValueError(
				f"{type(task_evaluator.task).__name__} reports no success criterion, so there "
				"is no success rate to record -- this stage needs a task whose steps report "
				"`success` (e.g. the ManiSkill tasks)."
			)

		# Per-episode rather than aggregated: "ever satisfied the success criterion" (not
		# "still satisfied on the final step", since ManiSkill recomputes success from the
		# current state every step rather than latching it), and env steps to the first
		# success, full episode length if it never came -- see `RealTaskEvaluator.__call__`.
		# `success_at_end` and `returns` are recorded alongside for the same reason: the mean
		# and sem `RealTaskEvaluator` also reports are recomputable from these distributions,
		# so only the per-episode numbers are kept.
		successes = [int(s) for s in info["episode_success_once_rate_distribution"]]
		success_at_end = [int(s) for s in info["episode_success_at_end_rate_distribution"]]
		time_to_success = [float(t) for t in info["episode_time_to_success_distribution"]]
		returns = [float(r) for r in info["episode_return_distribution"]]

		entry = {
			"num_trials": int(num_episodes),
			"replan_every": int(task_evaluator.cfg.replan_every),
			"gpu": latency_gpu,
			"successes": successes,
			"success_at_end": success_at_end,
			"time_to_success": time_to_success,
			"returns": returns,
			# Everything the evaluator returned (means, sems, distributions,
			# steps_needed_to_plan), and the settings it ran under, so nothing it reported is
			# lost and the entry says how it was produced.
			"evaluator_info": to_jsonable(info),
			"settings": {
				"rollout_gpu": gpu,
				"latency_run_name": latency_run_name,
				"plan_budget_s": latency,
				"num_recorded_latencies": len(latencies),
				"control_interval_s": float(control_interval),
				"num_envs": int(task_evaluator.num_envs),
				"max_episode_steps": task_evaluator.cfg.max_episode_steps,
				"interim_behaviour": task_evaluator.cfg.get("interim_behaviour"),
				"action_mode": task_evaluator.cfg.get("action_mode"),
				"save_video": bool(task_evaluator.cfg.save_video),
				"sim_backend": OmegaConf.select(cfg, "task.cfg.sim_backend"),
				"lighting": OmegaConf.select(cfg, "task.cfg.lighting"),
				"seed": seed,
				"config_name": HydraConfig.get().job.config_name,
				"feasible_configs": cfg.get("feasible_configs"),
			},
		}
		output_path = results_store.add_entry(run_name, planner_cfg, "success", entry)
		print(
			f"success_rate={np.mean(successes):.2f}, appended {len(successes)} trials to "
			f"{output_path}"
		)


if __name__ == "__main__":
	main()
