"""One-off: mean plan latency vs. mean time-to-success, Push Cube only.

A trimmed-down variant of `plot_mean_latency_vs_mean_time_to_success_json.py` for a
single figure rather than the usual 1x3-panel-per-preset grid:

  * Push Cube only, drawn as a single panel instead of one of three.
  * No "<preset> preset" suptitle and no panel title -- the figure is still for one
    preset's data (`--preset`, default "cool"), it just isn't labelled anywhere.
  * The y axis (time-to-success, seconds) is floored at 30 instead of 0, since
    every method's mean sits well above that on this task.
  * The method legend sits below the panel, in 2 rows of 3 rather than one long row.

Same underlying data and GPU-filtering as the grid version -- see that script's
docstring, or `results_store.py`, for the JSON schema and why latency and
time-to-success are both restricted to one device (`--gpu`, default "NVIDIA L40S").
Each point is one method's mean latency and mean time-to-success across every
(horizon, iterations) config cell that has data, with SEM error bars showing the
spread across configs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

TASK = "push_cube"

# label, filename prefix, colour, marker.  Colours/markers match plot_reachability.py
# so a method keeps its identity across figures.
METHODS = (
	("DINO-WM", "dino_wm", "#105ee8", "o"),
	("Sparse-Imagination", "sparse_imagination", "#099dab", "o"),
	("TC-WM", "tc_wm", "#00623d", "o"),
	("DINO-Bisim", "dino_bisim", "#3ebc11", "o"),
	("TD-MPC2", "tdmpc2", "#762e86", "o"),
	("TSWM (Ours)", "s2p", "#b291fd", "o"),
)

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

# All ManiSkill tasks run with task.cfg.frame_skip=1 (see
# s2p/configs/task/maniskill_*.yaml), and ManiSkill's control_freq is fixed at 20 Hz,
# so control_interval = 0.05 * frame_skip is 0.05s here -- see
# maniskill_task.py's get_control_interval(). The per-run JSON entries don't carry
# this (results_store.py's entry schema has no control_interval field), so it's
# hardcoded here; update it if push_cube's frame_skip ever changes.
CONTROL_INTERVAL = 0.05

# Lower bound of the y axis (seconds); every method's mean time-to-success on this
# task sits above this, so the floor is raised to spend more of the panel on the
# range that actually separates methods.
Y_MIN = 30.0
Y_MAX = 60.0

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
	than that module's own fixed `RESULTS_DIR`.
	"""
	path = results_dir / f"{run_name}.json"
	if not path.exists():
		return {}
	with open(path, "r") as f:
		raw = f.read()
	return json.loads(raw) if raw else {}


def load_cells(results_dir, method_prefix, preset, gpu):
	"""Return (latency_ms, time_to_success) grids for one method, one entry per
	(horizon, iterations) cell in HORIZONS x ITERATIONS order.

	A cell with no success entries recorded on `gpu`, or no latency entry recorded
	on `gpu` in the lighting-agnostic run those are read from, is left as NaN in
	both arrays rather than dropping the method.
	"""
	success_results = load_results(results_dir, f"{method_prefix}_{TASK}_{preset}")
	# Lighting-agnostic: plan latency doesn't depend on lighting_preset, so it's
	# recorded once per (method, task) and every preset's success reads from that
	# same file -- see analyze_latency.py.
	latency_results = load_results(results_dir, f"{method_prefix}_{TASK}")

	n = len(HORIZONS) * len(ITERATIONS)
	latency_ms = np.full(n, np.nan)
	tts = np.full(n, np.nan)

	idx = 0
	for horizon in HORIZONS:
		for iterations in ITERATIONS:
			key = planner_key(NUM_SAMPLES, NUM_ELITES, horizon, iterations)

			# Both stages are hardware-dependent, so both are pooled only from
			# entries recorded on `gpu` -- a run re-timed on different hardware
			# should not be silently averaged with this one.
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
				times = np.concatenate([e["time_to_success"] for e in success_entries])
				tts[idx] = times.mean() * CONTROL_INTERVAL
				latency_ms[idx] = float(np.mean(latencies)) * 1e3
			else:
				print(
					f"skipping {method_prefix}_{TASK}_{preset} h{horizon}i{iterations}: "
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


def plot(results_dir, preset, gpu):
	"""Load every method for Push Cube and return (fig, any_data)."""
	points = {}
	for _, method_prefix, _, _ in METHODS:
		cells = load_cells(results_dir, method_prefix, preset, gpu)
		if cells is None:
			continue
		latency_ms, tts = cells
		mean_latency, latency_sem = mean_and_sem(latency_ms)
		mean_tts, tts_sem = mean_and_sem(tts)
		points[method_prefix] = (mean_latency, latency_sem, mean_tts, tts_sem)

	if not points:
		return None, False

	latency_xlim = log_axis_limits([p[0] for p in points.values()])

	plt.rcParams.update(
		{
			"font.family": "sans-serif",
			"font.size": 15,
			"axes.linewidth": 0.8,
			"figure.dpi": 150,
		}
	)

	fig, ax = plt.subplots(figsize=(8, 3.8))

	for label, method_prefix, colour, marker in METHODS:
		point = points.get(method_prefix)
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
	ax.set_ylim(Y_MIN, Y_MAX)

	ax.set_xlabel("Mean Plan Latency (ms, log scale)")
	ax.set_ylabel("Mean Time to Success (s)")

	ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
	ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color("#c9c8c2")

	# Sizes the axes for the plot alone, before the legend goes on. Called after
	# the legend instead, tight_layout() recomputes the axes layout from scratch
	# without knowing about a legend placed via bbox_to_anchor, and pulls the axes
	# back down over it regardless of the anchor offset -- that was the actual
	# cause of the legend/x-label overlap, not the offset value.
	fig.tight_layout()

	method_handles = [
		Line2D(
			[], [], color=colour, linestyle="none", marker=marker,
			markersize=MARKER_SIZE * 0.7, markeredgecolor="white",
			markeredgewidth=MARKER_EDGEWIDTH, label=label,
		)
		for label, _, colour, marker in METHODS
	]
	# Anchored to the axes (not the figure) so the gap below the x axis is a fixed
	# small offset regardless of figure size. `savefig(..., bbox_inches="tight")`
	# below crops the saved figure to fit the legend, wherever it ends up.
	ax.legend(
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
		bbox_to_anchor=(0.5, -0.22),
		# 3 columns x 2 rows for the 6 methods, rather than one long row.
		ncol=3,
	)

	return fig, True


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=RESULTS_DIR,
		help="directory holding the <method>_push_cube.json and "
		"<method>_push_cube_<preset>.json result files (default: results_store.py's "
		"own results/ directory)",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=HERE,
		help="directory to write the figure into (default: this file's directory)",
	)
	parser.add_argument(
		"--preset",
		default="cool",
		help="lighting preset to plot (default: 'cool')",
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
		help="show the figure interactively instead of only saving it",
	)
	args = parser.parse_args()

	fig, has_data = plot(args.results_dir, args.preset, args.gpu)
	if not has_data:
		print(f"no data found for preset {args.preset!r}; nothing to plot")
		return
	out_path = args.out_dir / "mean_latency_vs_mean_tts_push_cube.png"
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	print(f"wrote {out_path}")
	if args.show:
		plt.show()
	plt.close(fig)


if __name__ == "__main__":
	main()
