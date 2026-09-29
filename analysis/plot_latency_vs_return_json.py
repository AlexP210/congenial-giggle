"""Scatter plan latency against episode return for several planners.

One figure per lighting preset ("cool", "default"), each a 1x3 grid with one panel
per task (push_cube, lift_peg, place_sphere). Within a panel, each planner
contributes the 3x3 (horizon x iteration) grid of measurements read from
`results_store.py`'s shared JSON files:

    results/<method>_<task>.json           latency entries -- lighting-agnostic, see
                                            analyze_latency.py
    results/<method>_<task>_<preset>.json  success entries, this preset only

Both are keyed within the file by the planner's own
`num_samples=<n>,num_elites=<e>,horizon=<h>,iterations=<i>` string -- see
results_store.py's module docstring for the exact schema. A (horizon, iterations)
cell missing from either file (not yet run, or still queued) is left out of that
planner's curve rather than dropping the whole method: results land one cell at a
time as array-job tasks complete, so a sweep still in progress should plot everything
that has landed so far.

Both latency and episode return are hardware-dependent (see results_store.latencies_for),
so entries recorded on a device other than `--gpu` are excluded from the pooled mean
rather than mixed in with it -- a cell only has both devices on record when a run was
re-timed on different hardware, and the two should not be averaged together.

The x axis is plan latency in milliseconds against episode return on the y axis. Each
planner's cloud of individual measurements sits behind its solid Pareto frontier
(here, "best" means highest return at or below a given latency).
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
PRESETS = ("cool", "default", "very-dim", "very-bright", "side")

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

# GPU that plan-latency and return measurements must have been recorded on to be
# pooled together -- see results_store.latencies_for. Vulcan's GPU nodes are 4x L40S.
GPU_NAME = "NVIDIA L40S"

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

# Fractional padding applied to a linear-scale axis's data range.
LINEAR_MARGIN = 0.08


def pareto_staircase(x, y):
	"""Return indices of the non-dominated points of (x, y), left to right, for a
	y that is better when *higher* (episode return) rather than lower.

	Sweeping left to right, a point is kept only if it beats every point at or to
	the left of it -- i.e. it is the running maximum of y.  Drawn with a
	zero-order hold, the kept points form a staircase that holds each level until
	a genuinely higher measurement arrives.  Ties are dropped: the leftmost point
	at a given y is the cheapest way to reach it.
	"""
	order = np.lexsort((-y, x))
	keep = []
	best = -np.inf
	for i in order:
		if y[i] > best:
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


def load_series(results_dir, method_prefix, task, preset, gpu):
	"""Return (latency_ms, episode_return, return_sem) grids for one
	method/task/preset, one entry per (horizon, iterations) cell in HORIZONS x
	ITERATIONS order (horizon outer, iterations inner -- matching the annotation
	labels below).

	A cell with no success entries recorded on `gpu`, or no latency entry recorded on
	`gpu` in the lighting-agnostic run those are read from, is left as NaN in every
	array rather than dropping the method: results land one array-job cell at a time,
	so a sweep still in progress should still plot whatever has completed. Returns
	None only if every cell is NaN, so a method/task that hasn't started yet drops out
	of the figure instead of contributing an all-NaN curve.
	"""
	success_results = load_results(results_dir, f"{method_prefix}_{task}_{preset}")
	# Lighting-agnostic: plan latency doesn't depend on lighting_preset, so it's
	# recorded once per (method, task) and every preset's success reads from that
	# same file -- see analyze_latency.py.
	latency_results = load_results(results_dir, f"{method_prefix}_{task}")

	n = len(HORIZONS) * len(ITERATIONS)
	latency_ms = np.full(n, np.nan)
	episode_return = np.full(n, np.nan)
	return_sem = np.full(n, np.nan)

	idx = 0
	for horizon in HORIZONS:
		for iterations in ITERATIONS:
			key = planner_key(NUM_SAMPLES, NUM_ELITES, horizon, iterations)

			# Both stages are hardware-dependent (episode return depends on the
			# device it ran on same as raw latency does), so both are pooled only
			# from entries recorded on `gpu` -- a run re-timed on different
			# hardware should not be silently averaged with this one.
			#
			# "returns" was added to the success entry schema after some results
			# were already recorded (see results_store.py's module docstring), so
			# older entries on `gpu` may be missing it -- those are excluded from
			# the pool rather than raising, same as a `gpu` mismatch.
			success_entries = [
				e for e in success_results.get(key, {}).get("success", [])
				if e["gpu"] == gpu
			]
			return_entries = [e for e in success_entries if "returns" in e]
			latency_entries = latency_results.get(key, {}).get("latency", [])
			latencies = [
				t for entry in latency_entries for t in entry["latencies"]
				if entry["gpu"] == gpu
			]

			if return_entries and latencies:
				# Pooled across every recorded invocation of this exact config --
				# a rerun, or several array-job attempts landing under the same
				# key -- not just the most recent one; see results_store.add_entry.
				returns = np.concatenate([e["returns"] for e in return_entries])
				episode_return[idx] = returns.mean()
				return_sem[idx] = returns.std(ddof=0) / np.sqrt(len(returns))

				latency_ms[idx] = float(np.mean(latencies)) * 1e3
			else:
				print(
					f"skipping {method_prefix}_{task}_{preset} h{horizon}i{iterations}: "
					f"missing returns or latency ({gpu!r}) entries"
				)

			idx += 1

	if np.isnan(latency_ms).all():
		return None
	return latency_ms, episode_return, return_sem


def log_axis_limits(values, margin=LOG_MARGIN):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values) & (values > 0)]
	return values.min() / margin, values.max() * margin


def linear_axis_limits(values, margin=LINEAR_MARGIN):
	"""Padded (lo, hi) limits spanning every finite value, unlike log_axis_limits
	this allows non-positive values through since episode return, unlike latency,
	is not bounded to be positive."""
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]
	lo, hi = values.min(), values.max()
	pad = (hi - lo) * margin if hi > lo else max(abs(hi), 1.0) * margin
	return lo - pad, hi + pad


def load_all_series(results_dir, preset, gpu):
	"""`{(method_prefix, task): (latency_ms, episode_return, return_sem)}` for
	every method/task with any data under one preset -- shared by the figure
	(which needs every task loaded up front to fix shared axis limits) and the
	LaTeX table."""
	series = {}
	for _, method_prefix, _, _ in METHODS:
		for _, task in TASKS:
			loaded = load_series(results_dir, method_prefix, task, preset, gpu)
			if loaded is not None:
				series[(method_prefix, task)] = loaded
	return series


def best_metric(values, sems, higher_is_better):
	"""(value, sem) of the best-performing (horizon, iterations) cell among the
	finite entries of `values`, or (nan, nan) if every cell is NaN (that
	method/task hasn't landed any results yet)."""
	values = np.asarray(values)
	finite = np.flatnonzero(np.isfinite(values))
	if finite.size == 0:
		return np.nan, np.nan
	pick = np.argmax if higher_is_better else np.argmin
	best_idx = finite[pick(values[finite])]
	return values[best_idx], sems[best_idx]


def format_cell(value, sem, decimals):
	if not np.isfinite(value):
		return "--"
	return f"${value:.{decimals}f} \\pm {sem:.{decimals}f}$"


def write_latex_table(series, out_path, preset):
	"""Write a Method x Task LaTeX table of best-config episode return (mean
	+/- SEM) to `out_path`, one row per method that has any data for this
	preset. Returns False (and writes nothing) if no method has any data."""
	rows = [
		(label, prefix) for label, prefix, _, _ in METHODS
		if any((prefix, task) in series for _, task in TASKS)
	]
	if not rows:
		return False

	lines = [
		"\\begin{table}[t]",
		"\\centering",
		"\\begin{tabular}{l" + "c" * len(TASKS) + "}",
		"\\toprule",
		"Method & " + " & ".join(title for title, _ in TASKS) + " \\\\",
		"\\midrule",
	]
	for label, prefix in rows:
		cells = []
		for _, task in TASKS:
			loaded = series.get((prefix, task))
			if loaded is None:
				cells.append("--")
				continue
			_, episode_return, return_sem = loaded
			value, sem = best_metric(episode_return, return_sem, higher_is_better=True)
			cells.append(format_cell(value, sem, decimals=2))
		lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
	lines += [
		"\\bottomrule",
		"\\end{tabular}",
		"\\caption{Episode return (mean $\\pm$ SEM) for the "
		f"best-performing planner configuration, {preset} preset.}}",
		f"\\label{{tab:latency_vs_return_{preset}}}",
		"\\end{table}",
	]
	out_path.write_text("\n".join(lines) + "\n")
	return True


def plot_preset(series, preset, annotate):
	"""Render the figure for one preset's already-loaded series (see
	`load_all_series`)."""
	all_latency = np.concatenate([s[0] for s in series.values()])
	latency_xlim = log_axis_limits(all_latency)
	all_returns = np.concatenate([s[1] for s in series.values()])
	return_ylim = linear_axis_limits(all_returns)

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
			latency_ms, episode_return, return_sem = loaded

			# Full detail against plan latency -- faded cloud of every
			# measurement plus its solid Pareto frontier.
			ax.errorbar(
				latency_ms,
				episode_return,
				yerr=return_sem,
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
			latency_frontier = frontier_line(latency_ms, episode_return)
			if latency_frontier is not None:
				line_x, line_y, mark_x, mark_y = latency_frontier
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
				for xi, yi, text in zip(latency_ms, episode_return, grid_labels[: len(latency_ms)]):
					if not (np.isfinite(xi) and np.isfinite(yi)):
						continue
					ax.annotate(
						text, (xi, yi), textcoords="offset points", xytext=(6, 5),
						fontsize=7, color="#5c5b55",
					)

		ax.set_xscale("log")
		ax.set_xlim(*latency_xlim)
		ax.set_ylim(*return_ylim)

		ax.set_title(title, fontsize=16, pad=20)
		ax.set_xlabel("Plan Latency (ms, log scale)")

		ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
		ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color("#c9c8c2")

	axes[0].set_ylabel("Episode Return")

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

	return fig


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
		help="only pool latency and return entries recorded on this device -- see "
		f"results_store.gpu_name (default: {GPU_NAME!r})",
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
		series = load_all_series(args.results_dir, preset, args.gpu)
		if not series:
			print(f"no data found for preset {preset!r}; skipping")
			continue

		fig = plot_preset(series, preset, args.annotate)
		out_path = args.out_dir / f"latency_vs_return_{preset}.png"
		fig.savefig(out_path, dpi=200, bbox_inches="tight")
		print(f"wrote {out_path}")
		if args.show:
			plt.show()
		plt.close(fig)

		tex_path = args.results_dir / f"latency_vs_return_{preset}.tex"
		if write_latex_table(series, tex_path, preset):
			print(f"wrote {tex_path}")


if __name__ == "__main__":
	main()
