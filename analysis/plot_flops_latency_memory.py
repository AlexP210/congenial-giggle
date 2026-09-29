"""Plan latency vs. planner horizon / iterations, and peak GPU memory, per method.

Reads the `analysis/results/<method>_<task>.json` files written by
`analyze_latency_flops_grid.py` (format: see `results_store.py`) and draws:

    left half  -- plan latency (ms) vs horizon and vs MPPI iterations, side by side,
                  one line per method, shared log y-axis
    right half -- one horizontal bar per method: peak GPU memory (GiB), linear axis

all in a single row at a 1:4 aspect ratio, panels lettered (A)-(C) left to right.

Every knob not being swept is held at the middle of its grid (`FIXED` below): the left
panel is the `iterations=3, num_samples=128, num_elites=16` slice, the right panel the
`horizon=3, ...` slice, and the memory bars are the single `FIXED` cell.

Memory gets a bar rather than a sweep because it does not depend on horizon or iterations
(under 2% across the whole 5x5 grid for every method on PushCube). It does scale with
`num_samples` -- roughly linearly for the DINO-based models (~3.9 / 7.5 / 14.9 GiB at
64 / 128 / 256 for DINO-WM) -- so the bars are only meaningful at the stated sample count.

Per cell:
  * latency   -- median over the timed repeats. The first repeat is dropped: it carries
                 the CUDA warm-up (e.g. 19.7 ms vs ~10.4 ms for the rest on S2P).
  * memory    -- median of `peak_allocated_bytes`, i.e. the total allocated at the peak of
                 a plan call (model weights included), which is what a GPU actually has to
                 hold -- not the increment over `baseline_allocated_bytes`.

There is no figure title and the held-fixed values are not drawn on the figure; state
them in the caption.

Example:

    python analysis/plot_flops_latency_memory.py --task push_cube
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# (file prefix, legend label, color, marker, linestyle). Labels, colors and order match
# plot_reachability.py and the plot_*_json.py scripts, so a method keeps its identity
# across figures. That palette pairs close hues (the two blues, the two greens), so marker
# + dash carry identity where color alone would not.
METHODS = [
	("dino_wm", "DINO-WM", "#105ee8", "^", "-."),
	("sparse_imagination", "Sparse-Imagination", "#099dab", "v", (0, (5, 1.5, 1, 1.5, 1, 1.5))),
	("tc_wm", "TC-WM", "#00623d", "P", (0, (1, 1))),
	("dino_bisim", "DINO-Bisim", "#3ebc11", "D", ":"),
	("tdmpc2", "TD-MPC2", "#762e86", "s", "--"),
	("s2p", "S2P (Ours)", "#b291fd", "o", "-"),
]

TASK_TITLES = {"push_cube": "PushCube", "pick_cube": "PickCube", "lift_peg": "LiftPeg"}

FIXED = {"num_samples": 128, "num_elites": 16, "horizon": 3, "iterations": 3}
# Full text width of a two-column paper; the height follows from the 1:4 aspect ratio.
FIG_WIDTH = 7.0

SWEEPS = [("horizon", "Planning horizon"), ("iterations", "MPPI iterations")]


def latency_ms(entry):
	"""Median plan latency (ms) of one cell, warm-up repeat dropped."""
	(latency,) = entry["latency"]
	return 1e3 * float(np.median(latency["latencies"][1:]))


def memory_gib(entry):
	"""Median peak allocated memory (GiB) of one cell."""
	(memory,) = entry["memory"]
	return float(np.median(memory["peak_allocated_bytes"])) / 2**30


def matching_cells(results, free_knob=None):
	"""Cells whose knobs all equal `FIXED`, except `free_knob`, sorted by `free_knob`."""
	cells = [
		entry for entry in results.values()
		if all(entry["planner"][k] == v for k, v in FIXED.items() if k != free_knob)
	]
	if free_knob is not None:
		cells.sort(key=lambda entry: entry["planner"][free_knob])
	return cells


def style_log_axis(ax):
	"""Log y-axis with a tick per decade, labelled in plain numbers (0.1, 1, 10, ...)."""
	ax.set_yscale("log")
	lo, hi = ax.get_ylim()
	ticks = [10.0**e for e in range(int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi))) + 1)]
	ax.yaxis.set_major_locator(FixedLocator(ticks))
	ax.yaxis.set_minor_formatter(NullFormatter())
	ax.set_yticklabels([f"{t:g}" for t in ticks])


def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--task", default="push_cube", choices=sorted(TASK_TITLES))
	parser.add_argument("--out", type=Path, default=None,
		help="output path without extension; .pdf and .png are both written "
		     "(default: analysis/latency_memory_<task>)")
	args = parser.parse_args()
	out = args.out or HERE / f"latency_memory_{args.task}"

	plt.rcParams.update({
		"font.family": "serif",
		"font.size": 9,
		"axes.titlesize": 10,
		"axes.labelsize": 9,
		"xtick.labelsize": 8,
		"ytick.labelsize": 8,
		"legend.fontsize": 8,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.edgecolor": "#555555",
		"axes.linewidth": 0.8,
		"xtick.color": "#555555",
		"ytick.color": "#555555",
		"axes.grid": True,
		"axes.axisbelow": True,
		"grid.color": "#e3e3e3",
		"grid.linewidth": 0.6,
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
	})

	results = {
		prefix: json.loads((RESULTS_DIR / f"{prefix}_{args.task}.json").read_text())
		for prefix, *_ in METHODS
	}

	# One row at 1:4. Two subfigures split it exactly in half -- latency lines on the left,
	# memory bars on the right -- however wide either side's tick labels are.
	fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH / 4), layout="constrained")
	line_fig, bar_fig = fig.subfigures(1, 2, wspace=0.02)
	line_axes = line_fig.subplots(1, len(SWEEPS), sharey=True)
	bar_ax = bar_fig.subplots()

	for ax, (knob, xlabel) in zip(line_axes, SWEEPS):
		for prefix, label, color, marker, linestyle in METHODS:
			cells = matching_cells(results[prefix], knob)
			xs = [entry["planner"][knob] for entry in cells]
			ax.plot(
				xs, [latency_ms(entry) for entry in cells], label=label, color=color,
				marker=marker, linestyle=linestyle, linewidth=1.5, markersize=4.5,
				markeredgecolor="white", markeredgewidth=0.6,
			)
		ax.set_xticks(xs)
		ax.set_xlabel(xlabel)
		ax.grid(axis="x", visible=False)
	line_axes[0].set_ylabel("Plan latency (ms)")
	for ax in line_axes:
		style_log_axis(ax)

	# Largest at the bottom, so the bars read as a ranking from the cheapest down.
	bars = []
	for prefix, label, color, *_ in METHODS:
		(cell,) = matching_cells(results[prefix])
		bars.append((memory_gib(cell), label, color))
	bars.sort(key=lambda bar: bar[0])
	positions = np.arange(len(bars))
	values = [value for value, _, _ in bars]
	bar_ax.barh(positions, values, height=0.62, color=[color for *_, color in bars],
		edgecolor="white", linewidth=1.0)
	bar_ax.set_yticks(positions, [label for _, label, _ in bars])
	bar_ax.invert_yaxis()
	bar_ax.tick_params(axis="y", length=0)
	bar_ax.spines["left"].set_visible(False)
	bar_ax.grid(axis="y", visible=False)
	bar_ax.set_xlabel("Peak GPU memory (GiB)")
	bar_ax.set_xlim(0, max(values) * 1.12)
	for position, value in zip(positions, values):
		text = f"{value * 1024:.0f} MiB" if value < 1 else f"{value:.2f} GiB"
		bar_ax.annotate(text, (value, position), xytext=(4, 0), textcoords="offset points",
			va="center", ha="left", fontsize=8, color="#333333")

	# Panel letters for referring to each plot from the text, as centred titles so they
	# sit above each panel rather than over its data.
	for letter, ax in zip("ABC", (*line_axes, bar_ax)):
		ax.set_title(f"({letter})", loc="center", fontweight="bold", fontsize=9, pad=4)

	handles, labels = line_axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="outside upper center", ncol=len(METHODS), frameon=False,
		handlelength=2.2, columnspacing=1.0, handletextpad=0.5)

	# No bbox_inches="tight": constrained layout already fits everything inside the figure,
	# and cropping would move the saved aspect ratio off 1:4.
	for ext in ("pdf", "png"):
		fig.savefig(f"{out}.{ext}", dpi=300)
		print(f"wrote {out}.{ext}")


if __name__ == "__main__":
	main()
