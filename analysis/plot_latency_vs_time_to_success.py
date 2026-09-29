"""Scatter planning compute against time-to-success for several planners.

One figure per lighting preset ("cool", "default"), each a 1x3 grid with one panel
per task (push_cube, lift_peg, place_sphere).  Within a panel, each planner
contributes the 3x3 (horizon x iteration) grid of measurements read from three
sibling directories of .npz files, all keyed by the same
`<method>_<task>_<preset>_H1-3_I1-3.npz` name:

    success_rate/<name>.npz   time_to_success_means, time_to_success_sem (env steps),
                              control_interval (seconds/step)
    flops/<name>.npz          plan_flops (FLOPs)
    latency/<name>.npz        plan_latency_means (seconds)

The bottom x axis is planning compute in GFLOPs; the top axis is a parallel scale
showing plan latency for the same grid of configurations, sharing the panel's y
axis. The two are measured independently (compute is a deterministic property of
the model, latency depends on the hardware it ran on), so they are drawn as two
separate frontiers -- solid against compute, dashed against latency -- rather than
a single line read against two rulers.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
GRID_SUFFIX = "H1-3_I1-3"

# label, filename prefix, colour, marker.  Colours/markers match plot_reachability.py
# so a method keeps its identity across figures.  TD-MPC2 and S2P are left out: no
# success_rate/flops/latency files exist for them under this naming scheme.
METHODS = (
	("DINO-WM", "dino_wm", "#105ee8", "o"),
	("Sparse-Imagination", "sparse_imagination", "#099dab", "o"),
	("TC-WM", "tc_wm", "#00623d", "o"),
	("DINO-Bisim", "dino_bisim", "#3ebc11", "o"),
)

# panel title, task slug (as it appears in the filenames).
TASKS = (
	("Push Cube", "push_cube"),
	("Lift Peg", "lift_peg"),
	("Place Sphere", "place_sphere"),
)

# Lighting presets the data was collected under; one figure is written per preset.
PRESETS = ("cool", "default")

# Grid axes of the sweep, matching analysis.py: rows are horizons, columns are
# planner iteration counts.  Only used for the optional per-point annotations.
HORIZONS = (1, 2, 3)
ITERATIONS = (1, 2, 3)

# Opacity of the individual measurements, which sit behind each planner's frontier.
POINT_ALPHA = 0.28

# Weight of the marks, scaled to the type sizes below rather than to matplotlib's
# defaults: at this figure's font size, default-weight lines and markers read as
# faint.
MARKER_SIZE = 7
MARKER_EDGEWIDTH = 0.9
LINEWIDTH = 2.0
LATENCY_LINEWIDTH = 1.4
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


def load_series(data_dir, method_prefix, task, preset):
	"""Return (gflops, time_to_success, tts_sem, latency_ms) for one method/task/preset.

	Returns None if any of the three source files is missing, so a sweep that
	has not been collected for every method yet just drops that method rather
	than failing the whole figure.
	"""
	name = f"{method_prefix}_{task}_{preset}_{GRID_SUFFIX}.npz"
	sr_path = data_dir / "success_rate" / name
	fl_path = data_dir / "flops" / name
	la_path = data_dir / "latency" / name
	if not (sr_path.exists() and fl_path.exists() and la_path.exists()):
		print(f"skipping {name}: missing source file(s)")
		return None

	with np.load(sr_path) as sr:
		control_interval = float(sr["control_interval"])
		tts = sr["time_to_success_means"].flatten() * control_interval
		tts_sem = sr["time_to_success_sem"].flatten() * control_interval
	with np.load(fl_path) as fl:
		gflops = fl["plan_flops"].flatten() / 1e9
	with np.load(la_path) as la:
		latency_ms = la["plan_latency_means"].flatten() * 1e3

	return gflops, tts, tts_sem, latency_ms


def log_axis_limits(values, margin=LOG_MARGIN):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values) & (values > 0)]
	return values.min() / margin, values.max() * margin


def plot_preset(data_dir, preset, annotate):
	"""Load every method/task under one preset and return (fig, any_data)."""
	# Load everything up front: the panels share y limits (time to success) and,
	# per axis, x limits (GFLOPs / latency) across tasks, which can only be set
	# once every task's range is known.
	series = {}
	for _, method_prefix, _, _ in METHODS:
		for _, task in TASKS:
			loaded = load_series(data_dir, method_prefix, task, preset)
			if loaded is not None:
				series[(method_prefix, task)] = loaded

	if not series:
		return None, False

	all_gflops = np.concatenate([s[0] for s in series.values()])
	all_latency = np.concatenate([s[3] for s in series.values()])

	gflops_xlim = log_axis_limits(all_gflops)
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
		ax2 = ax.twiny()

		for label, method_prefix, colour, marker in METHODS:
			loaded = series.get((method_prefix, task))
			if loaded is None:
				continue
			gflops, tts, tts_sem, latency_ms = loaded

			# Bottom axis: full detail against planning compute -- faded cloud of
			# every measurement plus its solid Pareto frontier.
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

			# Top axis: the same grid's frontier read against plan latency instead,
			# dashed and undecorated so it stays a secondary reference rather than
			# competing with the bottom axis's detail.
			latency_frontier = frontier_line(latency_ms, tts)
			if latency_frontier is not None:
				line_x, line_y, mark_x, mark_y = latency_frontier
				ax2.plot(
					line_x, line_y, color=colour, linewidth=LATENCY_LINEWIDTH,
					linestyle="--", drawstyle="steps-post", alpha=0.85, zorder=2,
				)
				ax2.plot(
					mark_x, mark_y, color=colour, linestyle="none", marker=marker,
					markersize=MARKER_SIZE * 0.7, markeredgecolor="white",
					markeredgewidth=MARKER_EDGEWIDTH, alpha=0.85, zorder=2,
				)

			if annotate:
				grid_labels = [f"h{h}i{i}" for h in HORIZONS for i in ITERATIONS]
				for xi, yi, text in zip(gflops, tts, grid_labels[: len(gflops)]):
					ax.annotate(
						text, (xi, yi), textcoords="offset points", xytext=(6, 5),
						fontsize=7, color="#5c5b55",
					)

		ax.set_xscale("log")
		ax2.set_xscale("log")
		ax.set_xlim(*gflops_xlim)
		ax2.set_xlim(*latency_xlim)
		ax.set_ylim(0.0, 60.0)

		ax.set_title(title, fontsize=16, pad=42)
		ax.set_xlabel("Plan Compute (GFLOPs, log scale)")
		ax2.set_xlabel("Plan Latency (ms, log scale)", fontsize=12, labelpad=8)

		ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
		ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
		ax2.tick_params(labelsize=TICK_LABELSIZE - 2, color="#c9c8c2")
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color("#c9c8c2")
		ax2.spines["top"].set_color("#c9c8c2")

	axes[0].set_ylabel("Time to Success (s)")

	method_handles = [
		Line2D(
			[], [], color=colour, linewidth=LINEWIDTH, marker=marker,
			markersize=MARKER_SIZE, markeredgecolor="white",
			markeredgewidth=MARKER_EDGEWIDTH, label=label,
		)
		for label, _, colour, marker in METHODS
	]
	style_handles = [
		Line2D([], [], color="#52514e", linewidth=LINEWIDTH, linestyle="-",
			   label="vs. compute (bottom axis)"),
		Line2D([], [], color="#52514e", linewidth=LATENCY_LINEWIDTH, linestyle="--",
			   label="vs. latency (top axis)"),
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
		bbox_to_anchor=(0.5, 0.04),
		ncol=len(METHODS),
	)
	fig.legend(
		handles=style_handles,
		title="Frontier",
		frameon=True,
		facecolor=LEGEND_FACECOLOR,
		edgecolor=LEGEND_EDGECOLOR,
		framealpha=1.0,
		borderpad=0.8,
		alignment="center",
		fontsize=LEGEND_FONTSIZE,
		title_fontproperties={"weight": "bold", "size": LEGEND_FONTSIZE + 1},
		loc="upper center",
		bbox_to_anchor=(0.5, -0.05),
		ncol=2,
	)
	fig.suptitle(f"{preset.capitalize()} preset", fontsize=18, y=1.04)
	fig.tight_layout(rect=(0, 0.06, 1, 1))

	return fig, True


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=HERE,
		help="directory holding the success_rate/, flops/ and latency/ subdirs "
		"(default: this file's directory)",
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
		fig, has_data = plot_preset(args.data_dir, preset, args.annotate)
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
