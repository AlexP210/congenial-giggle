"""Histogram the per-trial time-to-success distribution for every method and task.

One figure per (lighting preset, planner config) pair: a grid with one row per task
(push_cube, lift_peg, place_sphere) and one column per method (DINO-WM,
Sparse-Imagination, TC-WM, DINO-Bisim, TD-MPC2, Squeeze-to-Plan), so a single figure
shows how one fixed (horizon, iterations) config's raw trial-level time-to-success
values are shaped for every method/task combination -- not just their pooled mean,
which is all `plot_success_rate_vs_time_to_success_json.py` and
`plot_latency_vs_time_to_success_json.py` show. Data is read from
`results_store.py`'s shared JSON files:

    results/<method>_<task>_<preset>.json  success entries for this preset

keyed within the file by the planner's own
`num_samples=<n>,num_elites=<e>,horizon=<h>,iterations=<i>` string -- see
results_store.py's module docstring for the exact schema. An episode that never
succeeds contributes its full length to `time_to_success` (see analyze_success.py),
so a distribution skewed toward the right edge reflects failures, not just slow
successes.

Success rate/time-to-success are hardware-dependent (see
results_store.latencies_for: a success entry records which latency source it drew
its plan budget from), so entries recorded on a device other than `--gpu` are
excluded rather than pooled in with it.

A cell (method/task) with no success entries for this preset+config is left as an
empty, faded panel labeled "no data" rather than dropping the row/column, since
results land one array-job cell at a time and a sweep still in progress should still
render whatever has completed.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# label, filename prefix, colour -- columns, left to right. Colours match
# plot_reachability.py so a method keeps its identity across figures. TD-MPC2 and
# S2P are left out: no results files exist for them under this naming scheme.
METHODS = (
	("DINO-WM", "dino_wm", "#105ee8"),
	("Sparse-Imagination", "sparse_imagination", "#099dab"),
	("TC-WM", "tc_wm", "#00623d"),
	("DINO-Bisim", "dino_bisim", "#3ebc11"),
	("TD-MPC2", "tdmpc2", "#762e86"),
	("Squeeze-to-Plan (Ours)", "s2p", "#b291fd"),
)

# panel title, task slug (as it appears in the run names) -- rows, top to bottom.
TASKS = (
	("Push Cube", "push_cube"),
	("Lift Peg", "lift_peg"),
	("Place Sphere", "place_sphere"),
)

# Lighting presets the data was collected under; one set of figures is written per
# preset.
PRESETS = ("cool", "default", "very-bright", "very-dim", "side")

# Grid axes of the sweep, matching submit_jobs_s2p_analysis.sh: rows are horizons,
# columns are planner iteration counts. One figure is written per (horizon,
# iterations) pair.
HORIZONS = (1, 2, 3)
ITERATIONS = (1, 2, 3)

# num_samples/num_elites pairing the sweep was run with -- see
# submit_jobs_s2p_analysis.sh and submit_jobs_s2p_latency_flops.sh, which must agree
# on this pairing since the success stage looks latency/flops up by the resulting
# key, not by array index.
NUM_SAMPLES = 64
NUM_ELITES = 8

# GPU that success measurements must have drawn their plan budget from to be pooled
# together -- see results_store.latencies_for. Vulcan's GPU nodes are 4x L40S.
GPU_NAME = "NVIDIA L40S"

# All three ManiSkill tasks plotted here run with task.cfg.frame_skip=1 (see
# s2p/configs/task/maniskill_*.yaml), and ManiSkill's control_freq is fixed at 20 Hz,
# so control_interval = 0.05 * frame_skip is the same 0.05s for all of them -- see
# maniskill_task.py's get_control_interval(). The per-run JSON entries don't carry
# this (results_store.py's entry schema has no control_interval field), so it's
# hardcoded here; update it if a plotted task's frame_skip ever changes.
CONTROL_INTERVAL = 0.05

# Number of histogram bins, shared by every panel in a figure so bin widths --
# not just counts -- are comparable across methods and tasks.
NUM_BINS = 15

TICK_LABELSIZE = 10
MEAN_LINE_COLOR = "#3a3930"


def planner_key(num_samples, num_elites, horizon, iterations):
	"""Canonical key string for one grid cell, matching
	`results_store.planner_key`'s field order (num_samples, num_elites, horizon,
	iterations)."""
	return f"num_samples={num_samples},num_elites={num_elites},horizon={horizon},iterations={iterations}"


def load_results(results_dir, run_name):
	"""The `{config_key: {"planner": ..., <stage>: [...]}}` dict from
	`results/<run_name>.json`, or `{}` if that run has no results file yet.

	Mirrors `results_store.load`, but against a caller-supplied directory rather
	than that module's own fixed `RESULTS_DIR`, so this script can also be pointed
	at a copy of the results (e.g. synced from scratch).
	"""
	path = results_dir / f"{run_name}.json"
	if not path.exists():
		return {}
	with open(path, "r") as f:
		raw = f.read()
	return json.loads(raw) if raw else {}


def load_times(results_dir, method_prefix, task, preset, key, gpu):
	"""Per-trial time-to-success (seconds) for one method/task/preset/config cell,
	pooled across every recorded invocation of that exact config -- a rerun, or
	several array-job attempts landing under the same key -- not just the most
	recent one; see results_store.add_entry. Returns None if there are no success
	entries recorded on `gpu`."""
	success_results = load_results(results_dir, f"{method_prefix}_{task}_{preset}")
	success_entries = [
		e for e in success_results.get(key, {}).get("success", [])
		if e["gpu"] == gpu
	]
	if not success_entries:
		return None
	times = np.concatenate([e["time_to_success"] for e in success_entries])
	return times * CONTROL_INTERVAL


def plot_config(results_dir, preset, horizon, iterations, gpu):
	"""Load every method/task for one preset+config and return (fig, any_data)."""
	key = planner_key(NUM_SAMPLES, NUM_ELITES, horizon, iterations)

	times_by_cell = {}
	for _, method_prefix, _ in METHODS:
		for _, task in TASKS:
			times = load_times(results_dir, method_prefix, task, preset, key, gpu)
			if times is None:
				print(
					f"skipping {method_prefix}_{task}_{preset} h{horizon}i{iterations}: "
					f"missing success entries ({gpu!r})"
				)
				continue
			times_by_cell[(method_prefix, task)] = times

	if not times_by_cell:
		return None, False

	# Shared bin edges across every panel so bin widths -- not just counts -- are
	# comparable across methods and tasks; density normalizes for cells that pooled
	# a different number of trials (reruns land under the same key).
	all_times = np.concatenate(list(times_by_cell.values()))
	bins = np.linspace(0.0, all_times.max(), NUM_BINS + 1)

	plt.rcParams.update(
		{
			"font.family": "sans-serif",
			"font.size": 12,
			"axes.linewidth": 0.8,
			"figure.dpi": 150,
		}
	)

	fig, axes = plt.subplots(
		len(TASKS), len(METHODS),
		figsize=(3.1 * len(METHODS), 2.6 * len(TASKS)),
		sharex=True, sharey=True,
	)

	for row, (task_title, task) in enumerate(TASKS):
		for col, (method_label, method_prefix, colour) in enumerate(METHODS):
			ax = axes[row, col]
			times = times_by_cell.get((method_prefix, task))

			if times is None:
				ax.set_facecolor("#f5f4f0")
				ax.text(
					0.5, 0.5, "no data", transform=ax.transAxes,
					ha="center", va="center", fontsize=9, color="#a6a59c",
				)
			else:
				ax.hist(times, bins=bins, density=True, color=colour, alpha=0.75, zorder=3)
				ax.axvline(times.mean(), color=MEAN_LINE_COLOR, linewidth=1.2, zorder=4)
				ax.text(
					0.97, 0.93, f"n={len(times)}", transform=ax.transAxes,
					ha="right", va="top", fontsize=8, color="#5c5b55",
				)

			if row == 0:
				ax.set_title(method_label, fontsize=11, pad=8)
			if col == 0:
				ax.set_ylabel(task_title, fontsize=11)
			if row == len(TASKS) - 1:
				ax.set_xlabel("Time to Success (s)", fontsize=9)

			ax.grid(True, linewidth=0.5, color="#e3e2de", zorder=0)
			ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
			ax.set_axisbelow(True)
			for side in ("top", "right"):
				ax.spines[side].set_visible(False)
			for side in ("left", "bottom"):
				ax.spines[side].set_color("#c9c8c2")

	fig.suptitle(
		f"{preset.capitalize()} preset -- horizon={horizon}, iterations={iterations}",
		fontsize=15, y=1.02,
	)
	fig.tight_layout()

	return fig, True


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=RESULTS_DIR,
		help="directory holding the <method>_<task>_<preset>.json result files "
		"(default: results_store.py's own results/ directory)",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=HERE,
		help="directory to write one figure per preset/config into (default: this "
		"file's directory)",
	)
	parser.add_argument(
		"--presets",
		nargs="+",
		default=list(PRESETS),
		help=f"lighting presets to plot, one set of figures each (default: {list(PRESETS)})",
	)
	parser.add_argument(
		"--horizons",
		nargs="+",
		type=int,
		default=list(HORIZONS),
		help=f"planner horizons to plot, one figure each (default: {list(HORIZONS)})",
	)
	parser.add_argument(
		"--iterations",
		nargs="+",
		type=int,
		default=list(ITERATIONS),
		help=f"planner iteration counts to plot, one figure each (default: {list(ITERATIONS)})",
	)
	parser.add_argument(
		"--gpu",
		default=GPU_NAME,
		help="only pool success entries whose plan budget was drawn from this device -- "
		f"see results_store.gpu_name (default: {GPU_NAME!r})",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="show the figures interactively instead of only saving them",
	)
	args = parser.parse_args()

	for preset in args.presets:
		for horizon in args.horizons:
			for iterations in args.iterations:
				fig, has_data = plot_config(args.results_dir, preset, horizon, iterations, args.gpu)
				if not has_data:
					print(
						f"no data found for preset {preset!r} h{horizon}i{iterations}; skipping"
					)
					continue
				out_path = (
					args.out_dir
					/ f"time_to_success_dist_{preset}_h{horizon}i{iterations}.png"
				)
				fig.savefig(out_path, dpi=200, bbox_inches="tight")
				print(f"wrote {out_path}")
				if args.show:
					plt.show()
				plt.close(fig)


if __name__ == "__main__":
	main()
