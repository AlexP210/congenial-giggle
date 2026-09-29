#!/bin/bash

# Runs analysis/feasible_planner_configs.py for every method x task, writing one JSON per
# pair that lists the planner configs (horizon, iterations, num_samples, num_elites,
# replan_every) able to produce that task's action count within the time budget.
#
# Reads the measured latencies in analysis/results/<method>_<task>.json -- no GPU, no
# container, no model. The action count is fixed per task (NUM_ACTIONS below) and the time
# budget is TIME_BUDGET below. Any arguments are passed through to every call, e.g.
# --statistic max, or --horizon 3 4 5 to narrow the search.
#
# Output: analysis/feasible_configs/<budget>s/<method>_<task>.json, one directory per budget
# so runs with different budgets do not overwrite each other. A JSON that already exists is
# kept, not recomputed -- the submit_jobs_*_analysis.sh scripts read these files, so
# regenerating them changes what the next submission evaluates. Pass `redo` (or `--redo`)
# to recompute every pair; missing pairs are always computed. Either way, prints one summary
# row per method x task: feasible configs, and the estimated time to run a real_task_planning
# eval of every one of them (a lower bound -- see feasible_planner_configs.py).
#
# Example:
#
#   bash agents/squeeze2plan/analysis/run_feasible_planner_configs.sh                 # summary of what's on disk
#   bash agents/squeeze2plan/analysis/run_feasible_planner_configs.sh redo            # recompute everything
#   bash agents/squeeze2plan/analysis/run_feasible_planner_configs.sh redo --statistic max

set -euo pipefail

# if [ $# -lt 1 ]; then
#     echo "usage: $0 TIME_BUDGET_SECONDS [extra feasible_planner_configs.py args...]" >&2
#     exit 1
# fi
TIME_BUDGET=10

# `redo`/`--redo` is ours; everything else goes through to feasible_planner_configs.py.
REDO=false
PASSTHROUGH=()
for arg in "$@"; do
    if [ "$arg" = "redo" ] || [ "$arg" = "--redo" ]; then
        REDO=true
    else
        PASSTHROUGH+=("$arg")
    fi
done
set -- "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ANALYSIS_DIR}/feasible_configs/${TIME_BUDGET}s"

METHODS=(s2p tdmpc2 dino_wm dino_bisim sparse_imagination tc_wm)
TASKS=(push_cube pick_cube lift_peg)

# Planner values to search; each must be in the results files.
HORIZONS=(1 3 5)
ITERATIONS=(1 3 5)
REPLAN_EVERY=(1 3 5)

# Actions each task must produce within the budget.
declare -A NUM_ACTIONS=(
    [push_cube]=5
    [pick_cube]=6
    [lift_peg]=10
)

# Per-run tables go to /dev/null, and so do their "warning:" lines about unmeasured step cost
# or batched latency -- the summary marks those instead. Errors still reach stderr, and set -e
# stops the loop.
kept=()
for method in "${METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        output="${OUTPUT_DIR}/${method}_${task}.json"
        if [ "${REDO}" = false ] && [ -f "${output}" ]; then
            kept+=("${method}_${task}")
            continue
        fi
        python "${ANALYSIS_DIR}/feasible_planner_configs.py" \
            --method "${method}" \
            --task "${task}" \
            --time-budget "${TIME_BUDGET}" \
            --num-actions "${NUM_ACTIONS[$task]}" \
            --output "${output}" \
            --horizon "${HORIZONS[@]}" \
            --iterations "${ITERATIONS[@]}" \
            --replan-every "${REPLAN_EVERY[@]}" \
            "$@" \
            > /dev/null 2> >(grep -v '^warning:' >&2)
    done
done

if ((${#kept[@]})); then
    echo "Kept ${#kept[@]} existing JSON(s) in ${OUTPUT_DIR} (pass redo to recompute)."
    # Extra args only reach the pairs computed above, so a kept file may not reflect them.
    if (($#)); then
        echo "warning: '$*' was NOT applied to the kept files: ${kept[*]}" >&2
    fi
    echo
fi

# Summary table, read back from the JSONs just written.
python - "${ANALYSIS_DIR}" "${OUTPUT_DIR}" "${METHODS[*]}" "${TASKS[*]}" <<'PY'
import json
import sys
from pathlib import Path

analysis_dir, output_dir = sys.argv[1], Path(sys.argv[2])
methods, tasks = sys.argv[3].split(), sys.argv[4].split()
sys.path.insert(0, analysis_dir)
from feasible_planner_configs import format_duration

reports = {
	(method, task): json.loads((output_dir / f"{method}_{task}.json").read_text())
	for method in methods for task in tasks
}
total_s = sum(r["eval_estimate"]["total_eval_s_all_feasible"] for r in reports.values())

first = reports[methods[0], tasks[0]]
settings = first["eval_estimate"]
print(f"{first['time_budget_s']:g} s budget, {first['statistic']} latency on "
      f"{', '.join(first['gpu'])}; evals of {settings['num_episodes']} episodes x "
      f"{settings['max_episode_steps']} steps, {settings['num_envs']} env(s), "
      f"save_video={settings['save_video']}\n")


def eval_cell(report):
	"""Eval time, marked where part of it could not be measured (legend under the table)."""
	est = report["eval_estimate"]
	marks = ""
	if est["step_cost_source"] == "none":
		marks += "*"
	if "single_env_fallback" in est["plan_latency_sources"]:
		marks += "+"
	oom = f" ({est['num_oom']} OOM)" if est["num_oom"] else ""
	return f"{format_duration(est['total_eval_s_all_feasible'])}{marks}{oom}"

headers = ("method", "task", "feasible", "eval time")
rows = []
for method in methods:
	for i, task in enumerate(tasks):
		report = reports[method, task]
		rows.append((
			method if i == 0 else "",  # named once per block, so the column reads as groups
			task,
			f"{report['num_feasible']} / {report['num_candidates']}",
			eval_cell(report),
		))
footer = ("total", "", "", format_duration(total_s))

widths = [max(len(r[i]) for r in (headers, footer, *rows)) for i in range(len(headers))]
rule = "  ".join("-" * w for w in widths)
def line(row):
	return "  ".join(v.ljust(w) if i < 2 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths)))

print(line(headers))
print(rule)
for i, row in enumerate(rows):
	if i and i % len(tasks) == 0:
		print()
	print(line(row))
print(rule)
print(line(footer))

legend = []
if any(r["eval_estimate"]["step_cost_source"] == "none" for r in reports.values()):
	legend.append("*  no step cost measured: planning time only (run analyze_step_cost.py)")
if any("single_env_fallback" in r["eval_estimate"]["plan_latency_sources"] for r in reports.values()):
	legend.append("+  some configs lack batched latency at this num_envs: single-env latency used")
if any(r["eval_estimate"]["num_oom"] for r in reports.values()):
	legend.append("OOM  feasible configs that run out of GPU memory at this num_envs; not in the time")
if legend:
	print("\n" + "\n".join(legend))
print(f"\nJSONs in {output_dir}")
PY
