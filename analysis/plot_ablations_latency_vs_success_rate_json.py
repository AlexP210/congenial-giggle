"""Scatter plan latency against success rate for the S2P LiftPeg ablations.

The ablation counterpart of plot_latency_vs_success_rate_json.py: one figure with a panel per
lighting preset ("default" first), each showing full S2P alongside every ablation. Within a
panel, each variant contributes one point per feasible planner config that also has success
data, read from:

    feasible_configs/<budget>s/s2p_lift_peg.json         the feasible configs
    results/s2p_lift_peg.json                            latency entries, lighting-agnostic
    results/<variant>_lift_peg_<preset>.json             success entries, this preset only

The ablations share S2P's architecture and planner, so submit_jobs_s2p_ablations_analysis.sh
scores them over S2P's feasible configs and charges plans S2P's recorded latencies
(+latency_run_name=s2p_lift_peg); both are read from S2P's files here to match. The x position
of a config is therefore the same for every variant, and only its success rate differs. That
does not hold exactly for the fresh-encoder ablation, which runs the DINOv3 backbone in the
model rather than reading cached features.

Matching, GPU filtering, pooling and the Pareto frontier are as in
plot_latency_vs_success_rate_json.py, whose helpers this reuses.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_latency_vs_success_rate_json import (
	ERRORBAR_CAPSIZE,
	ERRORBAR_LINEWIDTH,
	FEASIBLE_DIR,
	GPU_NAME,
	HERE,
	LEGEND_EDGECOLOR,
	LEGEND_FACECOLOR,
	LEGEND_FONTSIZE,
	LINEWIDTH,
	MARKER_EDGEWIDTH,
	MARKER_SIZE,
	PANEL_ASPECT,
	POINT_ALPHA,
	PRESET_LABELS,
	PRESETS,
	RESULTS_DIR,
	TICK_LABELSIZE,
	TIME_BUDGET,
	best_metric,
	config_label,
	format_cell,
	frontier_line,
	load_feasible,
	load_results,
	log_axis_limits,
	planner_key,
)

TASK = "lift_peg"

# Run whose feasible configs and latencies every variant is scored against -- see the module
# docstring.
BASE_PREFIX = "s2p"

# label, success-file prefix, colour, marker. Full S2P keeps its colour from
# plot_latency_vs_success_rate_json.py.
METHODS = (
	("Squeeze-to-Plan (Full)", "s2p", "#b291fd", "o"),
	("Fresh Encoder", "s2p_ablate_fresh_encoder", "#e8743b", "o"),
	("No Dynamics Gradients", "s2p_ablate_no_dynamics_gradients", "#19a979", "o"),
	("No KL Gradients", "s2p_ablate_no_kl_gradients", "#ed4a7b", "o"),
	("No Reward Gradients", "s2p_ablate_no_reward_gradients", "#5899da", "o"),
)


def load_series(results_dir, feasible, latency_results, method_prefix, preset, gpu):
	"""(latency_ms, success_rate, success_rate_sem, labels) for one variant under one preset,
	one entry per feasible config with both success entries (for its `replan_every`) and
	latency entries recorded on `gpu`; None if no config has both."""
	success_results = load_results(results_dir, f"{method_prefix}_{TASK}_{preset}")

	latency_ms, success_rate, success_rate_sem, labels = [], [], [], []
	num_missing = 0
	for row in feasible:
		key = planner_key(row["num_samples"], row["num_elites"], row["horizon"], row["iterations"])
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

		successes = np.concatenate([e["successes"] for e in success_entries])
		p = successes.mean()
		latency_ms.append(float(np.mean(latencies)) * 1e3)
		success_rate.append(p)
		success_rate_sem.append(np.sqrt(p * (1.0 - p) / len(successes)))
		labels.append(config_label(row))

	if num_missing:
		print(
			f"{method_prefix}_{TASK}_{preset}: {num_missing} of {len(feasible)} feasible "
			f"configs have no success or latency ({gpu!r}) entries yet; left out"
		)
	if not labels:
		return None
	return np.array(latency_ms), np.array(success_rate), np.array(success_rate_sem), labels


def write_latex_table(series_by_preset, out_path):
	"""Variant x preset LaTeX table of best-config success rate (mean +/- SEM). Returns False
	(and writes nothing) if no variant has any data."""
	presets = list(series_by_preset)
	rows = [
		(label, prefix) for label, prefix, _, _ in METHODS
		if any(prefix in series for series in series_by_preset.values())
	]
	if not rows:
		return False

	lines = [
		"\\begin{table}[t]",
		"\\centering",
		"\\begin{tabular}{l" + "c" * len(presets) + "}",
		"\\toprule",
		"Method & " + " & ".join(PRESET_LABELS.get(p, p) for p in presets) + " \\\\",
		"\\midrule",
	]
	for label, prefix in rows:
		cells = []
		for preset in presets:
			loaded = series_by_preset[preset].get(prefix)
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
		"\\caption{Lift Peg ablations: success rate (mean $\\pm$ SEM) for the "
		"best-performing feasible planner configuration under each lighting preset.}",
		"\\label{tab:ablations_latency_vs_success_rate}",
		"\\end{table}",
	]
	out_path.write_text("\n".join(lines) + "\n")
	return True


def draw_panel(ax, series, annotate):
	"""Every variant's points and Pareto frontier under one preset."""
	for _, method_prefix, colour, marker in METHODS:
		loaded = series.get(method_prefix)
		if loaded is None:
			continue
		latency_ms, success_rate, success_rate_sem, labels = loaded

		ax.errorbar(
			latency_ms, success_rate, yerr=success_rate_sem, fmt=marker,
			markersize=MARKER_SIZE, markeredgecolor="white", markeredgewidth=MARKER_EDGEWIDTH,
			color=colour, ecolor=colour, elinewidth=ERRORBAR_LINEWIDTH,
			capsize=ERRORBAR_CAPSIZE, linestyle="none", alpha=POINT_ALPHA, zorder=3,
		)
		latency_frontier = frontier_line(latency_ms, success_rate)
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
	"""One figure with a panel per preset, in `series_by_preset`'s order, sharing both axes."""
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

	num_cols = len(series_by_preset)
	panel_width = 19 / 3
	fig, axes = plt.subplots(
		1, num_cols,
		figsize=(num_cols * panel_width, panel_width * PANEL_ASPECT + 1.5),
		sharex=True, sharey=True, squeeze=False,
	)

	for col, (preset, series) in enumerate(series_by_preset.items()):
		ax = axes[0, col]
		draw_panel(ax, series, annotate)

		ax.set_box_aspect(PANEL_ASPECT)
		ax.set_xscale("log")
		ax.set_xlim(*latency_xlim)
		ax.set_ylim(0.0, 1.05)
		ax.set_title(PRESET_LABELS.get(preset, preset), fontsize=16, pad=20)
		ax.set_xlabel("Plan Latency (ms, log scale)")

		ax.grid(True, which="both", linewidth=0.5, color="#e3e2de", zorder=0)
		ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color("#c9c8c2")

	axes[0, 0].set_ylabel("Lift Peg\nSuccess Rate")

	method_handles = [
		Line2D(
			[], [], color=colour, linewidth=LINEWIDTH, marker=marker,
			markersize=MARKER_SIZE, markeredgecolor="white",
			markeredgewidth=MARKER_EDGEWIDTH, label=label,
		)
		for label, _, colour, marker in METHODS
	]
	fig.tight_layout(rect=(0, 0.1, 1, 1))
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
		bbox_to_anchor=(0.5, 0.1),
		ncol=len(METHODS),
	)

	return fig


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--results-dir", type=Path, default=RESULTS_DIR,
		help="directory holding the result files (default: results_store.py's own results/)",
	)
	parser.add_argument(
		"--out-dir", type=Path, default=HERE,
		help="directory to write the figure into (default: this file's directory)",
	)
	parser.add_argument(
		"--presets", nargs="+", default=list(PRESETS),
		help=f"lighting presets to plot, one panel each, default first (default: {list(PRESETS)})",
	)
	parser.add_argument(
		"--feasible-dir", type=Path, default=FEASIBLE_DIR,
		help=f"directory holding the per-budget feasible-config reports (default: {FEASIBLE_DIR})",
	)
	parser.add_argument(
		"--time-budget", type=int, default=TIME_BUDGET,
		help=f"which budget's feasible configs to plot (default: {TIME_BUDGET})",
	)
	parser.add_argument(
		"--gpu", default=GPU_NAME,
		help=f"only pool latency and success entries recorded on this device (default: {GPU_NAME!r})",
	)
	parser.add_argument(
		"--annotate", action="store_true",
		help="label each point with its horizon, iterations, replan_every and num_samples",
	)
	parser.add_argument(
		"--show", action="store_true",
		help="show the figure interactively instead of only saving it",
	)
	args = parser.parse_args()

	feasible = load_feasible(args.feasible_dir / f"{args.time_budget}s", BASE_PREFIX, TASK)
	if feasible is None:
		print(f"no feasible report for {BASE_PREFIX}_{TASK}; nothing to plot")
		return
	latency_results = load_results(args.results_dir, f"{BASE_PREFIX}_{TASK}")

	presets = sorted(args.presets, key=lambda preset: preset != "default")
	series_by_preset = {}
	for preset in presets:
		series = {}
		for _, method_prefix, _, _ in METHODS:
			loaded = load_series(
				args.results_dir, feasible, latency_results, method_prefix, preset, args.gpu,
			)
			if loaded is not None:
				series[method_prefix] = loaded
		if not series:
			print(f"no data found for preset {preset!r}; skipping")
			continue
		series_by_preset[preset] = series

	if not series_by_preset:
		print("no data found for any preset; nothing to plot")
		return

	tex_path = args.results_dir / "ablations_latency_vs_success_rate.tex"
	if write_latex_table(series_by_preset, tex_path):
		print(f"wrote {tex_path}")

	fig = plot_presets(series_by_preset, args.annotate)
	for suffix in (".png", ".pdf"):
		out_path = args.out_dir / f"ablations_latency_vs_success_rate{suffix}"
		fig.savefig(out_path, dpi=200, bbox_inches="tight")
		print(f"wrote {out_path}")
	if args.show:
		plt.show()
	plt.close(fig)


if __name__ == "__main__":
	main()
