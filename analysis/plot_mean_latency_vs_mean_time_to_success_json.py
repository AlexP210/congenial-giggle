"""Scatter each method's mean plan latency against its mean time-to-success.

One figure per lighting preset ("cool", "default"), each a 1x3 grid with one panel
per task (push_cube, lift_peg, place_sphere). Unlike
`plot_latency_vs_time_to_success_json.py`, which draws one point per (horizon,
iterations) cell plus a Pareto frontier, this collapses each method's whole 3x3
(horizon x iteration) grid down to a single point: the mean latency and mean
time-to-success across every config cell that has data, with error bars showing the
spread across configs (SEM) rather than the spread across trials within a config.
That makes this the "one number per method" summary version of the same underlying
data read from `results_store.py`'s shared JSON files:

    results/<method>_<task>.json           latency entries -- lighting-agnostic, see
                                            analyze_latency.py
    results/<method>_<task>_<preset>.json  success entries, this preset only

Both are keyed within the file by the planner's own
`num_samples=<n>,num_elites=<e>,horizon=<h>,iterations=<i>` string -- see
results_store.py's module docstring for the exact schema. A (horizon, iterations)
cell missing from either file (not yet run, or still queued) is left out of the
average rather than dropping the whole method: results land one cell at a time as
array-job tasks complete, so a sweep still in progress should still average whatever
has landed so far.

Both latency and time-to-success are hardware-dependent (see
results_store.latencies_for), so entries recorded on a device other than `--gpu` are
excluded from the pooled mean rather than mixed in with it -- a cell only has both
devices on record when a run was re-timed on different hardware, and the two should
not be averaged together.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# label, filename prefix, colour, marker.  Colours/markers match plot_reachability.py
# so a method keeps its identity across figures.  TD-MPC2 and S2P are left out: no
# results files exist for them under this naming scheme.
METHODS = (
	("DINO-WM", "dino_wm", "#105ee8", "o"),
	("Sparse-Imagination", "sparse_imagination", "#099dab", "o"),
	("TC-WM", "tc_wm", "#00623d", "o"),
	("DINO-Bisim", "dino_bisim", "#3ebc11", "o"),
	("TD-MPC2", "tdmpc2", "#762e86", "o"),
	("Squeeze-to-Plan (Ours)", "s2p", "#b291fd", "o"),
)

# panel title, task slug (as it appears in the run names).
TASKS = (
	("Push Cube", "push_cube"),
	("Lift Peg", "lift_peg"),
	("Place Sphere", "place_sphere"),
)

# Lighting presets the data was collected under; one figure is written per preset.
PRESETS = ("cool", "default")

# Grid axes of the sweep, matching submit_jobs_s2p_analysis.sh: rows are horizons,
# columns are planner iteration counts. Only used to enumerate which config cells
# to average over.
HORIZONS = (1, 2, 3)
ITERATIONS = (1, 2, 3)

# num_samples/num_elites pairing the sweep was run with -- see
# submit_jobs_s2p_analysis.sh and submit_jobs_s2p_latency_flops.sh, which must agree
# on this pairing since the success stage looks latency/flops up by the resulting
# key, not by array index.
NUM_SAMPLES = 64
NUM_ELITES = 8

# GPU that plan-latency and success measurements must have been recorded on to be
# pooled together -- see results_store.latencies_for. Vulcan's GPU nodes are 4x L40S.
GPU_NAME = "NVIDIA L40S"

# All three ManiSkill tasks plotted here run with task.cfg.frame_skip=1 (see
# s2p/configs/task/maniskill_*.yaml), and ManiSkill's control_freq is fixed at 20 Hz,
# so control_interval = 0.05 * frame_skip is the same 0.05s for all of them -- see
# maniskill_task.py's get_control_interval(). The per-run JSON entries don't carry
# this the way the old .npz files did (results_store.py's entry schema has no
# control_interval field), so it's hardcoded here; update it if a plotted task's
# frame_skip ever changes.
CONTROL_INTERVAL = 0.05

# Weight of the marks, scaled to the type sizes below rather than to matplotlib's
# defaults: at this figure's font size, default-weight markers read as faint.
MARKER_SIZE = 14
MARKER_EDGEWIDTH = 1.2
ERRORBAR_LINEWIDTH = 1.4
ERRORBAR_CAPSIZE = 4

TICK_LABELSIZE = 13
LEGEND_FONTSIZE = 13

LEGEND_FACECOLOR = "#f2f1ec"
LEGEND_EDGECOLOR = "#c3c2b7"

# Multiplicative padding applied to a log-scale axis's data range.
LOG_MARGIN = 1.15


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


def load_cells(results_dir, method_prefix, task, preset, gpu):
	"""Return (latency_ms, time_to_success) grids for one method/task/preset, one
	entry per (horizon, iterations) cell in HORIZONS x ITERATIONS order.

	A cell with no success entries recorded on `gpu`, or no latency entry recorded
	on `gpu` in the lighting-agnostic run those are read from, is left as NaN in
	both arrays rather than dropping the method: results land one array-job cell at
	a time, so a sweep still in progress should still average whatever has landed
	so far.
	"""
	success_results = load_results(results_dir, f"{method_prefix}_{task}_{preset}")
	# Lighting-agnostic: plan latency doesn't depend on lighting_preset, so it's
	# recorded once per (method, task) and every preset's success reads from that
	# same file -- see analyze_latency.py.
	latency_results = load_results(results_dir, f"{method_prefix}_{task}")

	n = len(HORIZONS) * len(ITERATIONS)
	latency_ms = np.full(n, np.nan)
	tts = np.full(n, np.nan)

	idx = 0
	for horizon in HORIZONS:
		for iterations in ITERATIONS:
			key = planner_key(NUM_SAMPLES, NUM_ELITES, horizon, iterations)

			# Both stages are hardware-dependent (time-to-success measures how
			# many env steps a plan costs, which depends on the device it ran on
			# same as raw latency does), so both are pooled only from entries
			# recorded on `gpu` -- a run re-timed on different hardware should not
			# be silently averaged with this one.
			success_entries = [
				e for e in success_results.get(key, {}).get("success", [])
				if e["gpu"] == gpu
			]
			latency_entries = latency_results.get(key, {}).get("latency", [])
			latencies = [
				t for entry in latency_entries for t in entry["latencies"]
				if entry["gpu"] == gpu
			]

			if success_entries and latencies:
				# Pooled across every recorded invocation of this exact config --
				# a rerun, or several array-job attempts landing under the same
				# key -- not just the most recent one; see results_store.add_entry.
				times = np.concatenate([e["time_to_success"] for e in success_entries])
				tts[idx] = times.mean() * CONTROL_INTERVAL
				latency_ms[idx] = float(np.mean(latencies)) * 1e3
			else:
				print(
					f"skipping {method_prefix}_{task}_{preset} h{horizon}i{iterations}: "
					f"missing success or latency ({gpu!r}) entries"
				)

			idx += 1

	if np.isnan(latency_ms).all():
		return None
	return latency_ms, tts


def mean_and_sem(values):
	"""(mean, SEM) of the finite entries of `values` -- the spread across config
	cells, not across trials within a cell. SEM is 0 rather than NaN when only one
	cell is finite, so a method with a single completed config still gets a visible
	marker instead of an error bar matplotlib can't draw."""
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	mean = values.mean()
	sem = values.std(ddof=0) / np.sqrt(len(values)) if len(values) > 1 else 0.0
	return mean, sem


def log_axis_limits(values, margin=LOG_MARGIN):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values) & (values > 0)]
	return values.min() / margin, values.max() * margin


def plot_preset(results_dir, preset, gpu):
	"""Load every method/task under one preset and return (fig, any_data)."""
	# Load everything up front: the panels share x limits (mean latency) across
	# tasks, which can only be set once every task's range is known.
	points = {}
	for _, method_prefix, _, _ in METHODS:
		for _, task in TASKS:
			cells = load_cells(results_dir, method_prefix, task, preset, gpu)
			if cells is None:
				continue
			latency_ms, tts = cells
			mean_latency, latency_sem = mean_and_sem(latency_ms)
			mean_tts, tts_sem = mean_and_sem(tts)
			points[(method_prefix, task)] = (mean_latency, latency_sem, mean_tts, tts_sem)

	if not points:
		return None, False

	all_latency = [p[0] for p in points.values()]
	latency_xlim = log_axis_limits(all_latency)

	plt.rcParams.update(
		{
			"font.family": "sans-serif",
			"font.size": 15,
			"axes.linewidth": 0.8,
			"figure.dpi": 150,
		}
	)

	fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), sharey=True)

	for ax, (title, task) in zip(axes, TASKS):
		for label, method_prefix, colour, marker in METHODS:
			point = points.get((method_prefix, task))
			if point is None:
				continue
			mean_latency, latency_sem, mean_tts, tts_sem = point

			ax.errorbar(
				[mean_latency],
				[mean_tts],
				xerr=[latency_sem],
				yerr=[tts_sem],
				fmt=marker,
				markersize=MARKER_SIZE,
				markeredgecolor="white",
				markeredgewidth=MARKER_EDGEWIDTH,
				color=colour,
				ecolor=colour,
				elinewidth=ERRORBAR_LINEWIDTH,
				capsize=ERRORBAR_CAPSIZE,
				zorder=3,
			)

		ax.set_xscale("log")
		ax.set_xlim(*latency_xlim)
		ax.set_ylim(0.0, 60.0)

		ax.set_title(title, fontsize=16, pad=20)
		ax.set_xlabel("Mean Plan Latency (ms, log scale)")

		ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
		ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color("#c9c8c2")

	axes[0].set_ylabel("Mean Time to Success (s)")

	method_handles = [
		Line2D(
			[], [], color=colour, linestyle="none", marker=marker,
			markersize=MARKER_SIZE * 0.7, markeredgecolor="white",
			markeredgewidth=MARKER_EDGEWIDTH, label=label,
		)
		for label, _, colour, marker in METHODS
	]
	fig.legend(
		handles=method_handles,
		title="Method",
		frameon=True,
		facecolor=LEGEND_FACECOLOR,
		edgecolor=LEGEND_EDGECOLOR,
		framealpha=1.0,
		borderpad=0.8,
		alignment="center",
		fontsize=LEGEND_FONTSIZE,
		title_fontproperties={"weight": "bold", "size": LEGEND_FONTSIZE + 1},
		loc="upper center",
		bbox_to_anchor=(0.5, 0.0),
		ncol=len(METHODS),
	)
	fig.suptitle(f"{preset.capitalize()} preset", fontsize=18, y=1.02)
	fig.tight_layout(rect=(0, 0.05, 1, 1))

	return fig, True


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=RESULTS_DIR,
		help="directory holding the <method>_<task>.json and "
		"<method>_<task>_<preset>.json result files (default: results_store.py's "
		"own results/ directory)",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=HERE,
		help="directory to write one figure per preset into (default: this file's directory)",
	)
	parser.add_argument(
		"--presets",
		nargs="+",
		default=list(PRESETS),
		help=f"lighting presets to plot, one figure each (default: {list(PRESETS)})",
	)
	parser.add_argument(
		"--gpu",
		default=GPU_NAME,
		help="only pool latency and success entries recorded on this device -- see "
		f"results_store.gpu_name (default: {GPU_NAME!r})",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="show the figures interactively instead of only saving them",
	)
	args = parser.parse_args()

	for preset in args.presets:
		fig, has_data = plot_preset(args.results_dir, preset, args.gpu)
		if not has_data:
			print(f"no data found for preset {preset!r}; skipping")
			continue
		out_path = args.out_dir / f"mean_latency_vs_mean_time_to_success_{preset}.png"
		fig.savefig(out_path, dpi=200, bbox_inches="tight")
		print(f"wrote {out_path}")
		if args.show:
			plt.show()
		plt.close(fig)


if __name__ == "__main__":
	main()
