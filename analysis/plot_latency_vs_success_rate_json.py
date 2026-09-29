"""Scatter plan latency against success rate for several planners.

One figure with a row per lighting preset ("default" on top) and a column per task
(push_cube, lift_peg, pick_cube). Within a panel, each planner contributes one point per feasible
planner config -- every (horizon, iterations, num_samples, num_elites, replan_every)
row of its `feasible_planner_configs.py` report that also has success data -- read
from:

    feasible_configs/<budget>s/<method>_<task>.json  the feasible configs, see
                                                    run_feasible_planner_configs.sh
    results/<method>_<task>.json                     latency entries -- lighting-agnostic,
                                                    see analyze_latency.py
    results/<method>_<task>_<preset>.json            success entries, this preset only

Both are keyed within the file by the planner's own
`num_samples=<n>,num_elites=<e>,horizon=<h>,iterations=<i>` string -- see
results_store.py's module docstring for the exact schema. Success entries are matched
on `replan_every` too, since it is recorded on the entry rather than in the key. A
feasible config missing from either file (not yet run, or still queued) is left out of
that planner's curve rather than dropping the whole method, so a sweep still in
progress plots everything that has landed so far.

Both latency and success rate are hardware-dependent (see results_store.latencies_for),
so entries recorded on a device other than `--gpu` are excluded from the pooled mean
rather than mixed in with it -- a cell only has both devices on record when a run was
re-timed on different hardware, and the two should not be averaged together.

The x axis is plan latency in milliseconds against success rate on the y axis. Each
planner's cloud of individual measurements sits behind its solid Pareto frontier
(here, "best" means highest success rate at or below a given latency).
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
# so a method keeps its identity across figures.  The DINO methods are commented out
# until their feasible-config success sweeps have run.
METHODS = (
	# ("DINO-WM", "dino_wm", "#105ee8", "o"),
	# ("Sparse-Imagination", "sparse_imagination", "#099dab", "o"),
	# ("TC-WM", "tc_wm", "#00623d", "o"),
	# ("DINO-Bisim", "dino_bisim", "#3ebc11", "o"),
	("TD-MPC2", "tdmpc2", "#762e86", "o"),
	("Squeeze-to-Plan (Ours)", "s2p", "#b291fd", "o"),
)

# panel title, task slug (as it appears in the run names).
TASKS = (
	("Push Cube", "push_cube"),
	("Lift Peg", "lift_peg"),
	("Pick Cube", "pick_cube"),
)

# Lighting presets the data was collected under; one figure is written per preset.
PRESETS = (
	"default", 
	"bright-set-0.58125-1.9375",
	"bright-set-0.675-2.25",
	"bright-set-0.75-2.5",
	"warm-set-1.1", 
	"side", 
	"very-cool",
	"table-set-0.2",
	"object-hue-30",
	"object-hue-60"
)

# Row labels for the presets; a preset not listed here is labelled by its own name.
PRESET_LABELS = {
	"default": "Default",
	"bright-set-0.75-2.5": "Bright",
	"warm-set-1.1": "Warm",
	"object-hue-30": "Colour-Shift"
}

# Height / width of every panel's plotting area.
PANEL_ASPECT = 0.5

# Where run_feasible_planner_configs.sh writes its reports, one subdirectory per time
# budget; which configs get plotted is read from `<budget>s/<method>_<task>.json`.
FEASIBLE_DIR = HERE / "feasible_configs"
TIME_BUDGET = 10

# GPU that plan-latency and success measurements must have been recorded on to be
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


def pareto_staircase(x, y):
	"""Return indices of the non-dominated points of (x, y), left to right, for a
	y that is better when *higher* (success rate) rather than lower.

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


def load_feasible(feasible_dir, method_prefix, task):
	"""The `feasible` rows of `feasible_dir/<method>_<task>.json`, or None if there is no
	report for this method/task."""
	path = feasible_dir / f"{method_prefix}_{task}.json"
	if not path.exists():
		return None
	with open(path, "r") as f:
		return json.load(f)["feasible"]


def config_label(row):
	"""Short annotation for one feasible config: horizon, iterations, replan_every and
	num_samples."""
	return f"h{row['horizon']}i{row['iterations']}r{row['replan_every']}s{row['num_samples']}"


def load_series(results_dir, feasible_dir, method_prefix, task, preset, gpu):
	"""Return (latency_ms, success_rate, success_rate_sem, labels) for one
	method/task/preset: one entry per feasible config that has both success entries
	(for its `replan_every`) and latency entries recorded on `gpu`.

	Feasible configs without both are skipped rather than dropping the method, so a
	sweep still in progress plots whatever has completed. Returns None if there is no
	feasible report or none of its configs has data, so that method/task drops out of
	the figure.
	"""
	feasible = load_feasible(feasible_dir, method_prefix, task)
	if feasible is None:
		print(f"skipping {method_prefix}_{task}: no feasible report in {feasible_dir}")
		return None

	success_results = load_results(results_dir, f"{method_prefix}_{task}_{preset}")
	# Lighting-agnostic: plan latency doesn't depend on lighting_preset, so it's
	# recorded once per (method, task) and every preset's success reads from that
	# same file -- see analyze_latency.py.
	latency_results = load_results(results_dir, f"{method_prefix}_{task}")

	latency_ms, success_rate, success_rate_sem, labels = [], [], [], []
	num_missing = 0
	for row in feasible:
		key = planner_key(row["num_samples"], row["num_elites"], row["horizon"], row["iterations"])

		# Both stages are hardware-dependent (success rate depends on the device it
		# ran on same as raw latency does), so both are pooled only from entries
		# recorded on `gpu` -- a run re-timed on different hardware should not be
		# silently averaged with this one. `replan_every` is on the entry, not the key.
		success_entries = [
			e for e in success_results.get(key, {}).get("success", [])
			if e["gpu"] == gpu and e["replan_every"] == row["replan_every"]
		]
		latencies = [
			t for entry in latency_results.get(key, {}).get("latency", [])
			for t in entry["latencies"] if entry["gpu"] == gpu
		]
		if not (success_entries and latencies):
			num_missing += 1
			continue

		# Pooled across every recorded invocation of this exact config -- a rerun, or
		# a resumed sweep landing under the same key -- see results_store.add_entry.
		successes = np.concatenate([e["successes"] for e in success_entries])
		p = successes.mean()
		latency_ms.append(float(np.mean(latencies)) * 1e3)
		success_rate.append(p)
		success_rate_sem.append(np.sqrt(p * (1.0 - p) / len(successes)))
		labels.append(config_label(row))

	if num_missing:
		print(
			f"{method_prefix}_{task}_{preset}: {num_missing} of {len(feasible)} feasible "
			f"configs have no success or latency ({gpu!r}) entries yet; left out"
		)
	if not labels:
		return None
	return np.array(latency_ms), np.array(success_rate), np.array(success_rate_sem), labels


def log_axis_limits(values, margin=LOG_MARGIN):
	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values) & (values > 0)]
	return values.min() / margin, values.max() * margin


def load_all_series(results_dir, feasible_dir, preset, gpu):
	"""`{(method_prefix, task): (latency_ms, success_rate, success_rate_sem, labels)}`
	for every method/task with any data under one preset -- shared by the
	figure (which needs every task loaded up front to fix shared axis limits)
	and the LaTeX table."""
	series = {}
	for _, method_prefix, _, _ in METHODS:
		for _, task in TASKS:
			loaded = load_series(results_dir, feasible_dir, method_prefix, task, preset, gpu)
			if loaded is not None:
				series[(method_prefix, task)] = loaded
	return series


def best_metric(values, sems, higher_is_better):
	"""(value, sem) of the best-performing feasible config among the
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
	"""Write a Method x Task LaTeX table of best-config success rate (mean
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
			_, success_rate, success_rate_sem, _ = loaded
			value, sem = best_metric(success_rate, success_rate_sem, higher_is_better=True)
			cells.append(format_cell(value, sem, decimals=3))
		lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
	lines += [
		"\\bottomrule",
		"\\end{tabular}",
		"\\caption{Success rate (mean $\\pm$ SEM) for the "
		f"best-performing feasible planner configuration, {preset} preset.}}",
		f"\\label{{tab:latency_vs_success_rate_{preset}}}",
		"\\end{table}",
	]
	out_path.write_text("\n".join(lines) + "\n")
	return True


def write_mean_latex_table(series_by_preset, out_path):
	"""Write a Method x Task LaTeX table of best-config success rate averaged over the
	presets in `series_by_preset` -- the mean of the per-preset tables' cells, each
	preset's best config picked on its own -- to `out_path`.

	The SEM is that of a mean of independent estimates, sqrt(sum SEM_i^2) / K. A cell is
	"--" unless that method/task has data under every preset: a mean over only the
	presets that happen to have run is not comparable with a full one. Returns False
	(and writes nothing) if no cell has data under every preset.
	"""
	presets = list(series_by_preset)
	cells_by_method = {}
	for label, prefix, _, _ in METHODS:
		cells = []
		for _, task in TASKS:
			bests = [
				best_metric(series[(prefix, task)][1], series[(prefix, task)][2], higher_is_better=True)
				for series in series_by_preset.values() if (prefix, task) in series
			]
			if len(bests) < len(presets):
				cells.append(None)
				continue
			values, sems = np.array(bests).T
			cells.append((values.mean(), np.sqrt(np.sum(sems ** 2)) / len(presets)))
		if any(cell is not None for cell in cells):
			cells_by_method[label] = cells
	if not cells_by_method:
		return False

	preset_names = ", ".join(PRESET_LABELS.get(preset, preset) for preset in presets)
	lines = [
		"\\begin{table}[t]",
		"\\centering",
		"\\begin{tabular}{l" + "c" * len(TASKS) + "}",
		"\\toprule",
		"Method & " + " & ".join(title for title, _ in TASKS) + " \\\\",
		"\\midrule",
	]
	for label, cells in cells_by_method.items():
		formatted = [
			"--" if cell is None else format_cell(*cell, decimals=3) for cell in cells
		]
		lines.append(f"{label} & " + " & ".join(formatted) + " \\\\")
	lines += [
		"\\bottomrule",
		"\\end{tabular}",
		"\\caption{Success rate (mean $\\pm$ SEM) for the best-performing feasible planner "
		f"configuration, averaged over the {len(presets)} presets ({preset_names}).}}",
		"\\label{tab:latency_vs_success_rate_mean_over_presets}",
		"\\end{table}",
	]
	out_path.write_text("\n".join(lines) + "\n")
	return True


def draw_panel(ax, series, task, annotate):
	"""Every method's points and Pareto frontier for one task under one preset."""
	for label, method_prefix, colour, marker in METHODS:
		loaded = series.get((method_prefix, task))
		if loaded is None:
			continue
		latency_ms, success_rate, success_rate_sem, labels = loaded

		# Full detail against plan latency -- faded cloud of every
		# measurement plus its solid Pareto frontier.
		ax.errorbar(
			latency_ms,
			success_rate,
			yerr=success_rate_sem,
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
		# No frontier for a method that never succeeded here: a flat line along 0 reads as
		# a result rather than an absence of one, and hides the others' low ends.
		latency_frontier = (
			frontier_line(latency_ms, success_rate) if np.nanmax(success_rate) > 0 else None
		)
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
			for xi, yi, text in zip(latency_ms, success_rate, labels):
				if not (np.isfinite(xi) and np.isfinite(yi)):
					continue
				ax.annotate(
					text, (xi, yi), textcoords="offset points", xytext=(6, 5),
					fontsize=7, color="#5c5b55",
				)


def plot_presets(series_by_preset, annotate):
	"""One figure with a row per preset, in `series_by_preset`'s order, and a column per
	task. Every panel shares the x (latency) and y (success rate) axes."""
	all_latency = np.concatenate([
		s[0] for series in series_by_preset.values() for s in series.values()
	])
	latency_xlim = log_axis_limits(all_latency)

	plt.rcParams.update(
		{
			"font.family": "sans-serif",
			"font.size": 15,
			"axes.linewidth": 0.8,
			"figure.dpi": 150,
		}
	)

	num_rows = len(series_by_preset)
	panel_width = 19 / len(TASKS)
	fig, axes = plt.subplots(
		num_rows, len(TASKS),
		figsize=(19, num_rows * panel_width * PANEL_ASPECT + 1.5),
		sharex=True, sharey=True, squeeze=False,
	)

	for row, (preset, series) in enumerate(series_by_preset.items()):
		for col, (title, task) in enumerate(TASKS):
			ax = axes[row, col]
			draw_panel(ax, series, task, annotate)

			ax.set_box_aspect(PANEL_ASPECT)
			ax.set_xscale("log")
			ax.set_xlim(*latency_xlim)
			# Headroom above 1 so a frontier at 100% isn't drawn on the panel's edge, but
			# the axis itself -- spine and ticks -- stops at 1, the largest real value.
			ax.set_ylim(0.0, 1.05)
			ax.set_yticks(np.linspace(0.0, 1.0, 6))
			ax.spines["left"].set_bounds(0.0, 1.0)

			if row == 0:
				ax.set_title(title, fontsize=16, pad=20)
			if row == num_rows - 1:
				ax.set_xlabel("Plan Latency (ms, log scale)")

			ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
			ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
			ax.set_axisbelow(True)
			for side in ("top", "right"):
				ax.spines[side].set_visible(False)
			for side in ("left", "bottom"):
				ax.spines[side].set_color("#c9c8c2")

		axes[row, 0].set_ylabel(f"{PRESET_LABELS.get(preset, preset)}\nSuccess Rate")

	method_handles = [
		Line2D(
			[], [], color=colour, linewidth=LINEWIDTH, marker=marker,
			markersize=MARKER_SIZE, markeredgecolor="white",
			markeredgewidth=MARKER_EDGEWIDTH, label=label,
		)
		for label, _, colour, marker in METHODS
	]
	fig.tight_layout(rect=(0, 0.04, 1, 1))
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
		help="directory to write the figure into (default: this file's directory)",
	)
	parser.add_argument(
		"--presets",
		nargs="+",
		default=list(PRESETS),
		help="lighting presets to plot, one row each, default first "
		f"(default: {list(PRESETS)})",
	)
	parser.add_argument(
		"--feasible-dir",
		type=Path,
		default=FEASIBLE_DIR,
		help="directory holding run_feasible_planner_configs.sh's per-budget reports "
		f"(default: {FEASIBLE_DIR})",
	)
	parser.add_argument(
		"--time-budget",
		type=int,
		default=TIME_BUDGET,
		help="which budget's feasible configs to plot, i.e. the <budget>s subdirectory "
		f"(default: {TIME_BUDGET})",
	)
	parser.add_argument(
		"--gpu",
		default=GPU_NAME,
		help="only pool latency and success entries recorded on this device -- see "
		f"results_store.gpu_name (default: {GPU_NAME!r})",
	)
	parser.add_argument(
		"--annotate",
		action="store_true",
		help="label each point with its horizon, iterations, replan_every and num_samples",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="show the figure interactively instead of only saving it",
	)
	args = parser.parse_args()

	feasible_dir = args.feasible_dir / f"{args.time_budget}s"
	# "default" leads, so the stock lighting is the top row and the others read as
	# departures from it.
	presets = sorted(args.presets, key=lambda preset: preset != "default")
	series_by_preset = {}
	for preset in presets:
		series = load_all_series(args.results_dir, feasible_dir, preset, args.gpu)
		if not series:
			print(f"no data found for preset {preset!r}; skipping")
			continue
		series_by_preset[preset] = series

		tex_path = args.results_dir / f"latency_vs_success_rate_{preset}.tex"
		if write_latex_table(series, tex_path, preset):
			print(f"wrote {tex_path}")

	if not series_by_preset:
		print("no data found for any preset; nothing to plot")
		return

	tex_path = args.results_dir / "latency_vs_success_rate_mean_over_presets.tex"
	if write_mean_latex_table(series_by_preset, tex_path):
		print(f"wrote {tex_path}")

	fig = plot_presets(series_by_preset, args.annotate)
	for suffix in (".png", ".pdf"):
		out_path = args.out_dir / f"latency_vs_success_rate{suffix}"
		fig.savefig(out_path, dpi=200, bbox_inches="tight")
		print(f"wrote {out_path}")
	if args.show:
		plt.show()
	plt.close(fig)


if __name__ == "__main__":
	main()
