"""Scatter success rate against time-to-success for several planners.

One figure per lighting preset, each a 1x3 grid with one panel per task
(push_cube, lift_peg, place_sphere). Within a panel, each planner contributes one
dot per (horizon, iterations) config cell -- both axes are pooled per-cell
statistics (not raw per-trial measurements), so each dot already carries an error
bar: horizontal for the success-rate's binomial standard error, vertical for the
time-to-success SEM. Data is read from `results_store.py`'s shared JSON files:

    results/<method>_<task>_<preset>.json  success entries for this preset

keyed within the file by the planner's own
`num_samples=<n>,num_elites=<e>,horizon=<h>,iterations=<i>` string -- see
results_store.py's module docstring for the exact schema. Unlike
`plot_latency_vs_time_to_success_json.py` and `plot_latency_vs_success_rate_json.py`,
this plot needs no latency file: both success rate and time-to-success come from the
same per-cell success entries, so there is nothing lighting-agnostic to join in. A
(horizon, iterations) cell missing from the success file (not yet run, or still
queued) is left out of that planner's dots rather than dropping the whole method:
results land one cell at a time as array-job tasks complete, so a sweep still in
progress should plot everything that has landed so far.

Success rate is hardware-dependent (see results_store.latencies_for: a success entry
records which latency source it drew its plan budget from), so entries recorded on a
device other than `--gpu` are excluded from the pooled statistics rather than mixed
in with it.

The x axis is success rate against time-to-success on the y axis.
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
PRESETS = ("cool", "default", "very-bright", "very-dim", "side")

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

# Weight of the marks, scaled to the type sizes below rather than to matplotlib's
# defaults: at this figure's font size, default-weight lines and markers read as
# faint.
MARKER_SIZE = 9
MARKER_EDGEWIDTH = 1.0
ERRORBAR_LINEWIDTH = 1.3
ERRORBAR_CAPSIZE = 3.5
POINT_ALPHA = 0.85

TICK_LABELSIZE = 13
LEGEND_FONTSIZE = 13

LEGEND_FACECOLOR = "#f2f1ec"
LEGEND_EDGECOLOR = "#c3c2b7"


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
	"""Return (success_rate, success_rate_sem, tts, tts_sem) grids for one
	method/task/preset, one entry per (horizon, iterations) cell in HORIZONS x
	ITERATIONS order (horizon outer, iterations inner -- matching the annotation
	labels below).

	A cell with no success entries recorded on `gpu` is left as NaN in every array
	rather than dropping the method: results land one array-job cell at a time, so a
	sweep still in progress should still plot whatever has completed. Returns None
	only if every cell is NaN, so a method/task that hasn't started yet drops out of
	the figure instead of contributing an all-NaN curve.
	"""
	success_results = load_results(results_dir, f"{method_prefix}_{task}_{preset}")

	n = len(HORIZONS) * len(ITERATIONS)
	success_rate = np.full(n, np.nan)
	success_rate_sem = np.full(n, np.nan)
	tts = np.full(n, np.nan)
	tts_sem = np.full(n, np.nan)

	idx = 0
	for horizon in HORIZONS:
		for iterations in ITERATIONS:
			key = planner_key(NUM_SAMPLES, NUM_ELITES, horizon, iterations)

			# Success rate depends on the device the plan budget was drawn from same
			# as raw latency does, so it's pooled only from entries recorded on
			# `gpu` -- a run re-timed on different hardware should not be silently
			# averaged with this one.
			success_entries = [
				e for e in success_results.get(key, {}).get("success", [])
				if e["gpu"] == gpu
			]

			if success_entries:
				# Pooled across every recorded invocation of this exact config --
				# a rerun, or several array-job attempts landing under the same
				# key -- not just the most recent one; see results_store.add_entry.
				successes = np.concatenate([e["successes"] for e in success_entries])
				p = successes.mean()
				success_rate[idx] = p
				success_rate_sem[idx] = np.sqrt(p * (1.0 - p) / len(successes))

				times = np.concatenate([e["time_to_success"] for e in success_entries])
				tts[idx] = times.mean() * CONTROL_INTERVAL
				tts_sem[idx] = times.std(ddof=0) / np.sqrt(len(times)) * CONTROL_INTERVAL
			else:
				print(
					f"skipping {method_prefix}_{task}_{preset} h{horizon}i{iterations}: "
					f"missing success entries ({gpu!r})"
				)

			idx += 1

	if np.isnan(success_rate).all():
		return None
	return success_rate, success_rate_sem, tts, tts_sem


def plot_preset(results_dir, preset, gpu, annotate):
	"""Load every method/task under one preset and return (fig, any_data)."""
	series = {}
	for _, method_prefix, _, _ in METHODS:
		for _, task in TASKS:
			loaded = load_series(results_dir, method_prefix, task, preset, gpu)
			if loaded is not None:
				series[(method_prefix, task)] = loaded

	if not series:
		return None, False

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
			success_rate, success_rate_sem, tts, tts_sem = loaded

			# One dot per config cell -- both axes are already per-cell pooled
			# statistics, so the error bar on each dot is the whole story; there is
			# no raw-measurement cloud or frontier to draw behind it.
			ax.errorbar(
				success_rate,
				tts,
				xerr=success_rate_sem,
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

			if annotate:
				grid_labels = [f"h{h}i{i}" for h in HORIZONS for i in ITERATIONS]
				for xi, yi, text in zip(success_rate, tts, grid_labels[: len(success_rate)]):
					if not (np.isfinite(xi) and np.isfinite(yi)):
						continue
					ax.annotate(
						text, (xi, yi), textcoords="offset points", xytext=(6, 5),
						fontsize=7, color="#5c5b55",
					)

		ax.set_xlim(0.0, 1.05)
		ax.set_ylim(0.0, 60.0)

		ax.set_title(title, fontsize=16, pad=20)
		ax.set_xlabel("Success Rate")

		ax.grid(True, linewidth=0.5, color="#e3e2de", zorder=0)
		ax.tick_params(labelsize=TICK_LABELSIZE, color="#c9c8c2")
		ax.set_axisbelow(True)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color("#c9c8c2")

	axes[0].set_ylabel("Time to Success (s)")

	method_handles = [
		Line2D(
			[], [], color=colour, linestyle="none", marker=marker,
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
		help="directory holding the <method>_<task>_<preset>.json result files "
		"(default: results_store.py's own results/ directory)",
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
		help="only pool success entries whose plan budget was drawn from this device -- "
		f"see results_store.gpu_name (default: {GPU_NAME!r})",
	)
	parser.add_argument(
		"--annotate",
		action="store_true",
		help="label each dot with its horizon and iteration count",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="show the figures interactively instead of only saving them",
	)
	args = parser.parse_args()

	for preset in args.presets:
		fig, has_data = plot_preset(args.results_dir, preset, args.gpu, args.annotate)
		if not has_data:
			print(f"no data found for preset {preset!r}; skipping")
			continue
		out_path = args.out_dir / f"success_rate_vs_time_to_success_{preset}.png"
		fig.savefig(out_path, dpi=200, bbox_inches="tight")
		print(f"wrote {out_path}")
		if args.show:
			plt.show()
		plt.close(fig)


if __name__ == "__main__":
	main()
