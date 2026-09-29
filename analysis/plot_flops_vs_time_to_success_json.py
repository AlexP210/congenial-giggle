"""Scatter planning compute against time-to-success for several planners.

One figure per lighting preset ("cool", "default"), each a 1x3 grid with one panel
per task (push_cube, lift_peg, place_sphere). Within a panel, each planner
contributes the 3x3 (horizon x iteration) grid of measurements read from
`results_store.py`'s shared JSON files instead of the old success_rate/flops/latency
.npz grids:

    results/<method>_<task>.json           flops entries -- lighting-agnostic, see
                                            submit_jobs_s2p_latency_flops.sh
    results/<method>_<task>_<preset>.json  success entries, this preset only

Both are keyed within the file by the planner's own
`num_samples=<n>,num_elites=<e>,horizon=<h>,iterations=<i>` string -- see
results_store.py's module docstring for the exact schema. A (horizon, iterations)
cell missing from either file (not yet run, or still queued) is left out of that
planner's curve rather than dropping the whole method: unlike the old .npz grids,
which were written whole, these results land one cell at a time as array-job tasks
complete, so a sweep still in progress should plot everything that has landed so far.

The x axis is planning compute in GFLOPs -- a deterministic property of the model,
independent of what hardware it ran on -- against time-to-success on the y axis.
Each planner's cloud of individual measurements sits behind its solid Pareto
frontier.
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
# columns are planner iteration counts.
HORIZONS = (1, 2, 3)
ITERATIONS = (1, 2, 3)

# num_samples/num_elites pairing the sweep was run with -- see
# submit_jobs_s2p_analysis.sh and submit_jobs_s2p_latency_flops.sh, which must agree
# on this pairing since the success stage looks latency/flops up by the resulting
# key, not by array index.
NUM_SAMPLES = 64
NUM_ELITES = 8

# All three ManiSkill tasks plotted here run with task.cfg.frame_skip=1 (see
# s2p/configs/task/maniskill_*.yaml), and ManiSkill's control_freq is fixed at 20 Hz,
# so control_interval = 0.05 * frame_skip is the same 0.05s for all of them -- see
# maniskill_task.py's get_control_interval(). The new per-run JSON entries don't carry
# this the way the old .npz files did (results_store.py's entry schema has no
# control_interval field), so it's hardcoded here; update it if a plotted task's
# frame_skip ever changes.
CONTROL_INTERVAL = 0.05

# Opacity of the individual measurements, which sit behind each planner's frontier.
POINT_ALPHA = 0.28

# Weight of the marks, scaled to the type sizes below rather than to matplotlib's
# defaults: at this figure's font size, default-weight lines and markers read as
# faint.
MARKER_SIZE = 7
MARKER_EDGEWIDTH = 0.9
LINEWIDTH = 2.0
ERRORBAR_LINEWIDTH = 1.2
ERRORBAR_CAPSIZE = 3

TICK_LABELSIZE = 13
LEGEND_FONTSIZE = 13

LEGEND_FACECOLOR = "#f2f1ec"
LEGEND_EDGECOLOR = "#c3c2b7"

# Multiplicative padding applied to a log-scale axis's data range.
LOG_MARGIN = 1.15


def pareto_staircase(x, y):
	"""Return indices of the non-dominated points of (x, y), left to right.

	Sweeping left to right, a point is kept only if it beats every point at or to
	the left of it -- i.e. it is the running minimum of y.  Drawn with a
	zero-order hold, the kept points form a staircase that holds each level until
	a genuinely lower measurement arrives.  Ties are dropped: the leftmost point
	at a given y is the cheapest way to reach it.
	"""
	order = np.lexsort((y, x))
	keep = []
	best = np.inf
	for i in order:
		if y[i] < best:
			best = y[i]
			keep.append(i)
	return np.asarray(keep, dtype=int)


def frontier_line(x, y):
	"""Step-line coordinates tracing the Pareto frontier of (x, y).

	Returns (line_x, line_y, marker_x, marker_y), or None if there is nothing
	finite and positive to plot. The line is held flat out to the largest x
	measured, so it spans the whole range the planner was swept over.
	"""
	finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
	if not finite.any():
		return None
	step_x, step_y = x[finite], y[finite]
	keep = pareto_staircase(step_x, step_y)
	line_x, line_y = step_x[keep], step_y[keep]
	right_edge = step_x.max()
	if right_edge > line_x[-1]:
		line_x = np.append(line_x, right_edge)
		line_y = np.append(line_y, line_y[-1])
	return line_x, line_y, step_x[keep], step_y[keep]


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


def load_series(results_dir, method_prefix, task, preset):
	"""Return (gflops, time_to_success, tts_sem) grids for one method/task/preset,
	one entry per (horizon, iterations) cell in HORIZONS x ITERATIONS order
	(horizon outer, iterations inner -- matching the annotation labels below).

	A cell with no success entries, or no flops entry in the lighting-agnostic run
	those are read from, is left as NaN in every array rather than dropping the
	method: results land one array-job cell at a time, so a sweep still in
	progress should still plot whatever has completed. Returns None only if every
	cell is NaN, so a method/task that hasn't started yet drops out of the figure
	instead of contributing an all-NaN curve.
	"""
	success_results = load_results(results_dir, f"{method_prefix}_{task}_{preset}")
	# Lighting-agnostic: FLOPs don't depend on lighting_preset, so it's recorded
	# once per (method, task) and every preset's success reads from that same
	# file -- see submit_jobs_s2p_latency_flops.sh.
	flops_results = load_results(results_dir, f"{method_prefix}_{task}")

	n = len(HORIZONS) * len(ITERATIONS)
	gflops = np.full(n, np.nan)
	tts = np.full(n, np.nan)
	tts_sem = np.full(n, np.nan)

	idx = 0
	for horizon in HORIZONS:
		for iterations in ITERATIONS:
			key = planner_key(NUM_SAMPLES, NUM_ELITES, horizon, iterations)

			success_entries = success_results.get(key, {}).get("success", [])
			flops_entries = flops_results.get(key, {}).get("flops", [])

			if success_entries and flops_entries:
				# Pooled across every recorded invocation of this exact config --
				# a rerun, or several array-job attempts landing under the same
				# key -- not just the most recent one; see results_store.add_entry.
				times = np.concatenate([e["time_to_success"] for e in success_entries])
				tts[idx] = times.mean() * CONTROL_INTERVAL
				tts_sem[idx] = times.std(ddof=0) / np.sqrt(len(times)) * CONTROL_INTERVAL

				gflops[idx] = float(np.mean([e["flops"] for e in flops_entries])) / 1e9
			else:
				print(
					f"skipping {method_prefix}_{task}_{preset} h{horizon}i{iterations}: "
					"missing success or flops entries"
				)

			idx += 1

	if np.isnan(gflops).all():
		return None
	return gflops, tts, tts_sem


def log_axis_limits(values, margin=LOG_MARGIN):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values) & (values > 0)]
	return values.min() / margin, values.max() * margin


def plot_preset(results_dir, preset, annotate):
	"""Load every method/task under one preset and return (fig, any_data)."""
	# Load everything up front: the panels share y limits (time to success) and x
	# limits (GFLOPs) across tasks, which can only be set once every task's range
	# is known.
	series = {}
	for _, method_prefix, _, _ in METHODS:
		for _, task in TASKS:
			loaded = load_series(results_dir, method_prefix, task, preset)
			if loaded is not None:
				series[(method_prefix, task)] = loaded

	if not series:
		return None, False

	all_gflops = np.concatenate([s[0] for s in series.values()])
	gflops_xlim = log_axis_limits(all_gflops)

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
			loaded = series.get((method_prefix, task))
			if loaded is None:
				continue
			gflops, tts, tts_sem = loaded

			# Full detail against planning compute -- faded cloud of every
			# measurement plus its solid Pareto frontier.
			ax.errorbar(
				gflops,
				tts,
				yerr=tts_sem,
				fmt=marker,
				markersize=MARKER_SIZE,
				markeredgecolor="white",
				markeredgewidth=MARKER_EDGEWIDTH,
				color=colour,
				ecolor=colour,
				elinewidth=ERRORBAR_LINEWIDTH,
				capsize=ERRORBAR_CAPSIZE,
				linestyle="none",
				alpha=POINT_ALPHA,
				zorder=3,
			)
			compute_frontier = frontier_line(gflops, tts)
			if compute_frontier is not None:
				line_x, line_y, mark_x, mark_y = compute_frontier
				ax.plot(
					line_x, line_y, color=colour, linewidth=LINEWIDTH,
					drawstyle="steps-post", zorder=4,
				)
				ax.plot(
					mark_x, mark_y, color=colour, linestyle="none", marker=marker,
					markersize=MARKER_SIZE, markeredgecolor="white",
					markeredgewidth=MARKER_EDGEWIDTH, zorder=5,
				)

			if annotate:
				grid_labels = [f"h{h}i{i}" for h in HORIZONS for i in ITERATIONS]
				for xi, yi, text in zip(gflops, tts, grid_labels[: len(gflops)]):
					if not (np.isfinite(xi) and np.isfinite(yi)):
						continue
					ax.annotate(
						text, (xi, yi), textcoords="offset points", xytext=(6, 5),
						fontsize=7, color="#5c5b55",
					)

		ax.set_xscale("log")
		ax.set_xlim(*gflops_xlim)
		ax.set_ylim(0.0, 60.0)

		ax.set_title(title, fontsize=16, pad=20)
		ax.set_xlabel("Plan Compute (GFLOPs, log scale)")

		ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
		ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color("#c9c8c2")

	axes[0].set_ylabel("Time to Success (s)")

	method_handles = [
		Line2D(
			[], [], color=colour, linewidth=LINEWIDTH, marker=marker,
			markersize=MARKER_SIZE, markeredgecolor="white",
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
		"--annotate",
		action="store_true",
		help="label each point with its horizon and iteration count",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="show the figures interactively instead of only saving them",
	)
	args = parser.parse_args()

	for preset in args.presets:
		fig, has_data = plot_preset(args.results_dir, preset, args.annotate)
		if not has_data:
			print(f"no data found for preset {preset!r}; skipping")
			continue
		out_path = args.out_dir / f"flops_vs_time_to_success_{preset}.png"
		fig.savefig(out_path, dpi=200, bbox_inches="tight")
		print(f"wrote {out_path}")
		if args.show:
			plt.show()
		plt.close(fig)


if __name__ == "__main__":
	main()
