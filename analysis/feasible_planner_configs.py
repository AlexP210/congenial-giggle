"""Which planner configs can produce N actions within a time budget, from measured latencies.

Reads `analysis/results/<method>_<task>.json` (written by `analyze_latency_flops_grid.py`,
format in `results_store.py`) -- no GPU, no model -- and, for every measured
(horizon, iterations, num_samples, num_elites) cell crossed with every `replan_every` from 1
up to that cell's horizon, works out how long producing `--num-actions` actions takes:

    plans needed  = ceil(num_actions / replan_every)
    planning time = plans needed x per-plan latency

which is how the task evaluators consume a plan: each `model.plan` call is followed by
`replan_every` env steps executed open-loop from it (see `SimulationTaskEvaluator`), and
`replan_every` is capped at the horizon because a plan is only `horizon` actions long -- the
same cap `sweep_planners.py` applies. A config is feasible when its planning time is at most
`--time-budget`.

The per-plan latency is one timed repeat of `PlanLatencyEvaluator`: encoding one observation
plus one `model.plan` call, CUDA-synchronised, at a single env -- exactly what each replan
costs. The first repeat of each cell is dropped (it carries the CUDA warm-up), and the rest
are reduced by `--statistic`: `median` for the typical case, `max` for a worst-case bound.
Only planning compute is counted, not env stepping or anything else in the control loop.

The configs searched are only what the results file holds (on the current sweep: horizon and
iterations 1-5, populations 64/8, 128/16, 256/32), timed on whatever GPU the file records.
`--horizon`, `--iterations`, `--num-samples` and `--num-elites` narrow that to the listed
values; each defaults to every value in the file, and naming one the file lacks is an error.

Each feasible config also gets an estimate of how long a `real_task_planning` eval of it takes
(`RealTaskEvaluator`, as `analyze_success.py` launches it):

    episode steps        = time budget / control interval
    steps_needed_to_plan = ceil(mean recorded 1-env latency / control interval)
    plans per episode    = ceil(episode steps / (steps_needed_to_plan + replan_every))
    eval time            = batches x ( reset
                                     + plans per episode x plan latency at num_envs
                                     + episode steps x step cost )

Episodes last `--time-budget` seconds of task time -- the same interval the feasibility
check used -- i.e. the evaluator's `max_episode_steps` is `time budget / 0.05 s` primitive
steps (200 for 10 s), and they run their full length, since every eval config here uses a
`-v1.1` task id, which never terminates early, in `ceil(num_episodes / num_envs)` batches.
The step budget copies `analyze_success.py` exactly: the mean over every recorded repeat,
converted at ManiSkill's 20 Hz x `--eval-frame-skip`. A replan fires every
`steps_needed_to_plan + replan_every` steps, first at step 0.

The eval's own latency measurement is assumed bypassed, since the L40S latencies already
size the plan budget. As written, `RealTaskEvaluator.__call__` still times 100 plans with
its `PlanLatencyEvaluator` before the first episode, even when `steps_needed_to_plan` is
supplied; until that is skipped, add 100 x latency per eval.

Step cost and reset come from `analyze_step_cost.py`'s measurements in
`results/step_cost/<method>_<task>.json`: the entries on the same GPU as the latencies, at
`--eval-num-envs`, with the same `--eval-save-video`, pooled, median per step and per reset.
`--eval-step-overhead-ms` overrides the measured step cost. With neither, the step cost is 0
and the estimate is a planning-only lower bound -- the 36 S2PSweep1 evals in s2p_sweep/
suggest that misses ~27 ms/step at 50 envs, for S2P.

Plan latency at `num_envs` is the single-env latency when `--eval-num-envs` is 1. Above 1 it
is the `batched_latency` entry `analyze_step_cost.py` recorded for that cell at that
`num_envs`, reduced by `--statistic`; a cell recorded as out of memory is marked `eval_oom`,
gets no eval time and is left out of the total (it is still real-time feasible -- the plan
budget is single-env either way -- it just cannot be evaluated at that `num_envs` on that
GPU). A cell with no such entry falls back to its single-env latency, which understates a
batched plan. Each report records which of these sources it used.

Model loading and env construction are not counted.

Example -- which S2P configs produce 6 actions within 10 s on PushCube:

    python analysis/feasible_planner_configs.py --method s2p --task push_cube \\
        --time-budget 10 --num-actions 6
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import results_store

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

STATISTICS = {"median": np.median, "max": np.max}

# ManiSkill's control interval at frame_skip=1 (`ManiSkillTask.get_control_interval`).
PRIMITIVE_CONTROL_INTERVAL_S = 0.05

# Planner fields a search can be restricted to, and the command-line flag for each.
PLANNER_FILTERS = {
	"horizon": "--horizon",
	"iterations": "--iterations",
	"num_samples": "--num-samples",
	"num_elites": "--num-elites",
}


def plan_latency_s(entry, statistic):
	"""Per-plan latency (s) of one results cell, warm-up repeat dropped."""
	(latency,) = entry["latency"]
	return float(STATISTICS[statistic](latency["latencies"][1:])), latency["gpu"]


def budget_latency_s(entry):
	"""The latency `analyze_success.py` sizes the plan budget from: mean of every repeat."""
	(latency,) = entry["latency"]
	return float(np.mean(latency["latencies"]))


def eval_plan_latency(entry, gpu, num_envs, statistic):
	"""(plan latency at `num_envs` in s or None if it ran out of memory, source) for one cell."""
	single_env, _ = plan_latency_s(entry, statistic)
	if num_envs == 1:
		return single_env, "single_env"
	# `.get`: results files written before `batched_latency` existed have no such key.
	matching = [
		e for e in entry.get("batched_latency", [])
		if e["gpu"] == gpu and e["num_envs"] == num_envs
	]
	if not matching:
		return single_env, "single_env_fallback"
	if any(e.get("oom", False) for e in matching):
		return None, "measured_oom"
	latencies = [t for e in matching for t in e["latencies"]]
	return float(STATISTICS[statistic](latencies)), "measured"


def measured_step_cost(run_name, gpus, num_envs, save_video):
	"""(median step s, median reset s, number of entries pooled) from `analyze_step_cost.py`,
	or None if nothing on record matches this GPU, `num_envs` and `save_video`."""
	matching = [
		e for e in results_store.load_step_cost(run_name)
		if e["gpu"] in gpus and e["num_envs"] == num_envs and e["save_video"] == save_video
	]
	if not matching:
		return None
	step_times = [t for e in matching for t in e["step_times"]]
	reset_times = [t for e in matching for t in e["reset_times"]]
	return float(np.median(step_times)), float(np.median(reset_times)), len(matching)


def eval_estimate(row, eval_cfg):
	"""Estimated `real_task_planning` wall-clock for one candidate row; see module docstring."""
	control_interval = PRIMITIVE_CONTROL_INTERVAL_S * eval_cfg["frame_skip"]
	episode_steps = math.ceil(eval_cfg["max_episode_steps"] / eval_cfg["frame_skip"])
	steps_needed_to_plan = math.ceil(row["budget_latency_s"] / control_interval)
	plans_per_episode = math.ceil(episode_steps / (steps_needed_to_plan + row["replan_every"]))
	num_batches = math.ceil(eval_cfg["num_episodes"] / eval_cfg["num_envs"])
	eval_s = None
	if row["eval_plan_latency_s"] is not None:
		eval_s = num_batches * (
			eval_cfg["reset_s"]
			+ plans_per_episode * row["eval_plan_latency_s"]
			+ episode_steps * eval_cfg["step_cost_s"]
		)
	return {
		"steps_needed_to_plan": steps_needed_to_plan,
		"plans_per_episode": plans_per_episode,
		"eval_s": eval_s,
		"eval_oom": row["eval_plan_latency_s"] is None,
	}


def restrict(results, allowed, parser):
	"""The results cells whose planner fields are all in `allowed` ({field: values or None}).

	A field left at None is unrestricted. Asking for a value the file never measured is an
	error rather than a silently smaller search.
	"""
	for field, values in allowed.items():
		if values is None:
			continue
		measured = sorted({entry["planner"][field] for entry in results.values()})
		missing = sorted(set(values) - set(measured))
		if missing:
			parser.error(f"{PLANNER_FILTERS[field]} {missing} not in the results file; "
			             f"measured values are {measured}")
	return {
		key: entry for key, entry in results.items()
		if all(values is None or entry["planner"][field] in values
		       for field, values in allowed.items())
	}


def candidates(results, num_actions, statistic, replan_every_values, eval_cfg):
	"""One row per (results cell, replan_every <= horizon) pair."""
	rows = []
	for entry in results.values():
		planner = entry["planner"]
		latency, gpu = plan_latency_s(entry, statistic)
		eval_latency, eval_latency_source = eval_plan_latency(
			entry, gpu, eval_cfg["num_envs"], statistic
		)
		for replan_every in replan_every_values or range(1, planner["horizon"] + 1):
			if replan_every > planner["horizon"]:
				continue
			num_plans = math.ceil(num_actions / replan_every)
			row = {
				**planner,
				"replan_every": replan_every,
				"num_plans": num_plans,
				"latency_s": latency,
				"total_s": num_plans * latency,
				"budget_latency_s": budget_latency_s(entry),
				"eval_plan_latency_s": eval_latency,
				"eval_plan_latency_source": eval_latency_source,
				"gpu": gpu,
			}
			rows.append({**row, **eval_estimate(row, eval_cfg)})
	return rows


def format_duration(seconds):
	"""Seconds, minutes or hours, whichever keeps the number readable."""
	if seconds < 60:
		return f"{seconds:.1f} s"
	if seconds < 3600:
		return f"{seconds / 60:.1f} min"
	return f"{seconds / 3600:.2f} h"


def print_table(rows, time_budget):
	columns = [
		("horizon", "H", "{}"),
		("iterations", "iters", "{}"),
		("num_samples", "samples", "{}"),
		("num_elites", "elites", "{}"),
		("replan_every", "replan", "{}"),
		("num_plans", "plans", "{}"),
		("latency_s", "ms/plan", "{:.1f}"),
		("total_s", "total (s)", "{:.3f}"),
	]
	cells = [
		[fmt.format(row[key] * 1e3 if key == "latency_s" else row[key]) for key, _, fmt in columns]
		+ [
			f"{time_budget - row['total_s']:.3f}",
			str(row["steps_needed_to_plan"]),
			str(row["plans_per_episode"]),
			"OOM" if row["eval_oom"] else format_duration(row["eval_s"]),
		]
		for row in rows
	]
	headers = [header for _, header, _ in columns] + ["slack (s)", "eval steps/plan", "eval plans/ep", "eval time"]
	widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
	print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
	for c in cells:
		print("  ".join(v.rjust(w) for v, w in zip(c, widths)))


def main():
	parser = argparse.ArgumentParser(
		description=__doc__.splitlines()[0],
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("--method", required=True,
		help="results file prefix, e.g. s2p, tdmpc2, dino_wm")
	parser.add_argument("--task", required=True, help="e.g. push_cube, pick_cube, lift_peg")
	parser.add_argument("--time-budget", type=float, required=True,
		help="seconds available to produce the actions")
	parser.add_argument("--num-actions", type=int, required=True,
		help="number of actions that must be produced within the budget")
	parser.add_argument("--replan-every", type=int, nargs="+", default=None,
		help="replan intervals to consider (default: every value from 1 to each cell's horizon)")
	for field, flag in PLANNER_FILTERS.items():
		parser.add_argument(flag, type=int, nargs="+", default=None, dest=field,
			help=f"only consider these {field} values (default: every value in the results file)")
	parser.add_argument("--statistic", choices=sorted(STATISTICS), default="median",
		help="how the timed repeats of a cell are reduced to one per-plan latency")
	parser.add_argument("--show-infeasible", action="store_true",
		help="also list the configs that do not fit the budget")
	parser.add_argument("--output", type=Path, default=None,
		help="also write the query and the feasible configs to this JSON file")
	# The real_task_planning eval being estimated; defaults are real_task_planning.yaml's,
	# except the episode length, which is the time budget (see eval_cfg below).
	parser.add_argument("--eval-num-episodes", type=int, default=10)
	parser.add_argument("--eval-num-envs", type=int, default=1)
	parser.add_argument("--eval-frame-skip", type=int, default=1)
	parser.add_argument("--eval-save-video", action=argparse.BooleanOptionalAction, default=True,
		help="whether the eval records video (real_task_planning.yaml: yes); selects the "
		     "matching step-cost measurement, since it adds a render per step")
	parser.add_argument("--eval-step-overhead-ms", type=float, default=None,
		help="per-env-step wall-clock beyond planning, overriding the measured step cost "
		     "(default: analyze_step_cost.py's measurement, or 0 if there is none)")
	args = parser.parse_args()
	if args.num_actions < 1 or args.time_budget <= 0:
		parser.error("--num-actions must be >= 1 and --time-budget > 0")

	run_name = f"{args.method}_{args.task}"
	path = RESULTS_DIR / f"{run_name}.json"
	results = json.loads(path.read_text())
	latency_gpus = sorted({entry["latency"][0]["gpu"] for entry in results.values()})

	measured = measured_step_cost(run_name, latency_gpus, args.eval_num_envs, args.eval_save_video)
	if args.eval_step_overhead_ms is not None:
		step_cost_s = args.eval_step_overhead_ms * 1e-3
		reset_s = measured[1] if measured else 0.0
		step_cost_source = "flag"
	elif measured:
		step_cost_s, reset_s, _ = measured
		step_cost_source = "measured"
	else:
		step_cost_s, reset_s = 0.0, 0.0
		step_cost_source = "none"
		print(f"warning: no step cost measured for {run_name} on {latency_gpus} at "
		      f"{args.eval_num_envs} env(s), save_video={args.eval_save_video} -- eval times "
		      f"count planning only (run analyze_step_cost.py)", file=sys.stderr)
	eval_cfg = {
		"num_episodes": args.eval_num_episodes,
		"num_envs": args.eval_num_envs,
		# Episodes last the time budget: `max_episode_steps` is in primitive steps, which the
		# round() guards against 10 / 0.05 landing a hair above 200.
		"max_episode_steps": math.ceil(round(args.time_budget / PRIMITIVE_CONTROL_INTERVAL_S, 6)),
		"frame_skip": args.eval_frame_skip,
		"save_video": args.eval_save_video,
		"step_cost_s": step_cost_s,
		"reset_s": reset_s,
		"step_cost_source": step_cost_source,
	}
	results = restrict(results, {field: getattr(args, field) for field in PLANNER_FILTERS}, parser)
	if not results:
		# Possible even with every value measured: num_samples and num_elites are swept as
		# pairs (64/8, 128/16, ...), not crossed, so e.g. 64 samples with 16 elites is empty.
		parser.error("no measured planner config matches every restriction together")
	rows = candidates(results, args.num_actions, args.statistic, args.replan_every, eval_cfg)
	# Cheapest first within each group, so the head of the table is the most headroom and
	# the tail the most expensive configs that still fit.
	rows.sort(key=lambda row: (row["total_s"], -row["horizon"], -row["iterations"]))
	feasible = [row for row in rows if row["total_s"] <= args.time_budget]
	infeasible = [row for row in rows if row["total_s"] > args.time_budget]
	latency_sources = sorted({row["eval_plan_latency_source"] for row in feasible})
	if "single_env_fallback" in latency_sources:
		print(f"warning: some feasible configs of {run_name} have no batched latency at "
		      f"{args.eval_num_envs} envs -- their eval planning time uses the single-env "
		      f"latency, which understates it (run analyze_step_cost.py with the grid)",
		      file=sys.stderr)

	gpus = sorted({row["gpu"] for row in rows})
	print(f"{path.name}: {args.num_actions} actions within {args.time_budget:g} s, "
	      f"{args.statistic} per-plan latency on {', '.join(gpus)}")
	print(f"{len(feasible)} of {len(rows)} (planner config, replan_every) pairs fit.\n")
	evaluable = [row for row in feasible if not row["eval_oom"]]
	total_eval_s = sum(row["eval_s"] for row in evaluable)
	if feasible:
		print_table(feasible, args.time_budget)
		oom = len(feasible) - len(evaluable)
		print(f"\nEvaluating {len(evaluable)} feasible configs with real_task_planning "
		      f"({args.eval_num_episodes} episodes, {args.eval_num_envs} env(s), "
		      f"{eval_cfg['max_episode_steps']}-step ({args.time_budget:g} s) episodes, "
		      f"step cost {step_cost_s * 1e3:.1f} ms [{step_cost_source}], "
		      f"plan latency [{', '.join(latency_sources)}]): {format_duration(total_eval_s)}"
		      + (f"; {oom} more run out of memory at {args.eval_num_envs} envs" if oom else ""))
	if args.show_infeasible and infeasible:
		print(f"\nDo not fit ({len(infeasible)}):")
		print_table(infeasible, args.time_budget)

	if args.output is not None:
		fields = (
			*PLANNER_FILTERS, "replan_every", "num_plans", "latency_s", "total_s",
			"eval_plan_latency_s", "eval_plan_latency_source",
			"steps_needed_to_plan", "plans_per_episode", "eval_s", "eval_oom",
		)
		report = {
			"method": args.method,
			"task": args.task,
			"results_file": path.name,
			"num_actions": args.num_actions,
			"time_budget_s": args.time_budget,
			"statistic": args.statistic,
			"gpu": gpus,
			# null means unrestricted: every value in the results file was searched.
			"restrictions": {
				**{field: getattr(args, field) for field in PLANNER_FILTERS},
				"replan_every": args.replan_every,
			},
			"num_candidates": len(rows),
			"num_feasible": len(feasible),
			# How each eval time was built; see the module docstring for the sources.
			"eval_estimate": {
				**eval_cfg,
				"plan_latency_sources": latency_sources,
				"num_oom": len(feasible) - len(evaluable),
				"total_eval_s_all_feasible": total_eval_s,
			},
			"feasible": [{field: row[field] for field in fields} for row in feasible],
		}
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(json.dumps(report, indent=2) + "\n")
		print(f"\nwrote {args.output}")


if __name__ == "__main__":
	main()
