"""Iso-budget planner-config frontiers for several methods.

For a given planning-latency budget, which MPPI configurations (planning horizon
x optimization iterations) can each method actually run in time?  Horizon and
iterations are integers and latency rises monotonically with both, so the set
reachable within a budget is a lower-left "staircase" region: everything
below-and-left of a method's line is reachable within that budget.  Each
frontier steps through the furthest reachable tested configuration at every
iteration level.

Colour identifies the method, line style the budget, and each method carries the
budgets it is worth drawing at -- see RUNS.  This is the multi-method
version of `trc_reachable_config_contours.py`, reading the sweeps written by
`collect_latency_data.py` rather than loose `.npy` files:

    analysis/planner_latencies/<run_name>.npz

Each of those carries its own `horizons` and `iterations` axes, so runs swept
over different grids can be drawn together; the faint dots mark the union of the
tested configurations.

Frontiers can and do coincide -- two methods that afford exactly the same
configurations draw exactly the same staircase -- so a stretch of line several
methods share is drawn as that many lines running side by side, rather than once
per method with the last one winning. The lanes straddle the true position, so a
line nobody shares is drawn exactly on the configurations it describes.

A dot marks a configuration measured to be the frontier at its iteration level;
a line that simply ends carries no dot, because the sweep is what ended, not the
frontier. That happens at both ends. On the right, a method that can already
afford the widest horizon tested has its frontier somewhere off the grid, so the
line runs out to the last measured configuration and stops rather than turning up
the last column as though the budget had stopped it there. At the top, a line
reaching the highest iteration count tested carries on above the plot; only one
that ends lower down -- with nothing at all reachable in the row above it -- has
really ended, and keeps its dot. Which frontiers ran off the grid is reported on
stdout, since the plot cannot show it.

Run from anywhere -- paths resolve relative to this file:

    python plot_reachability.py
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "planner_latencies"

# label, run name (the `runner.cfg.run_name` the sweep was collected under),
# colour, and the planning budgets (ms) to draw that method at.  Colours match
# `plot_latency_vs_time_to_success.py`, so a method keeps its identity across the
# two figures.
#
# The budgets are per method rather than global: a budget only says something
# about a method whose frontier it actually crosses, and one that every
# configuration fits inside (or none does) just adds a line along the edge of the
# grid.  Give a method several to show how its frontier moves with the budget.
# palette = ["#105ee8", "#099dab", "#00623d", "#3ebc11", "#762e86", "#b291fd"]
RUNS = (
	("DINO-WM", "DINOWM_PushCube", "#105ee8", (1000,)),
	("Sparse-Imagination", "LatencySparseImagination_PushCube", "#099dab", (1000,)),
	("TC-WM", "LatencyTCWM_PushCube", "#00623d", (1000,)),
	("DINO-Bisim", "LatencyDINOBisim_PushCube", "#3ebc11", (1000,)),
	("TD-MPC2", "Latency_TDMPC2_PushCube", "#762e86", (30,)),
	("S2P (Ours)", "LatencyS2P_PushCube", "#b291fd", (30,)),
)

# Line style per budget.  Every budget named in RUNS needs an entry: a budget
# that quietly fell back to a default style would be indistinguishable from
# whichever budget already owns that style, in the plot and in the legend alike.
BUDGET_STYLE = {10: ":", 30: "--", 1000: "-", 50: "-."}

# View window.  Frontiers and tested-config dots are always computed over the
# full swept grid; these only zoom the axes to the lower-left corner.  Clamped
# to the data, so values at or above the sweep size show everything.
N_HORIZONS = 10
N_ITERATIONS = 10

# Legends sit to the right of the axes: `LEGEND_X` is their left edge in axes
# coordinates, so > 1 is outside the frame. Smaller than the tick labels because
# they are a key rather than part of the plot.
LEGEND_X = 1.04
LEGEND_FONTSIZE = 14

# The box drawn around each legend: a light grey panel with a slightly darker
# rule, borrowed from the grid and tick colours so the key reads as part of the
# figure's furniture rather than as another mark on it.
LEGEND_FACECOLOR = "#f2f1ec"
LEGEND_EDGECOLOR = "#c3c2b7"

# Frontier line width, and the gap left between two of them running side by
# side, both in points. Every line is this wide wherever it goes: methods
# sharing a segment run beside each other rather than dividing one band between
# them, so a line's weight never says anything about how many methods are on it.
# An upper bound rather than a fixed size -- both are scaled down together if
# the figure's most-shared segment would otherwise stack wider than a dot.
LINEWIDTH = 3.0
LANE_GAP = 0.6

# The dot marking a furthest-reachable configuration: matplotlib's `s`, which is
# the square of its diameter in points (measured, not assumed -- `s` is
# documented as an area, but a scatter circle of `s` spans sqrt(s) points), and
# the white ring drawn over its rim to separate it from the line underneath.
#
# The visible colour of a dot is therefore sqrt(MARKER_SIZE) - MARKER_EDGEWIDTH
# across, and that is the ceiling a stack of lanes is scaled to fit: matching
# the full dot instead would leave the stack visibly fatter than the colour it
# runs through.
#
# Methods that share a dot draw concentric ones inside it: a corner is where a
# horizontal segment meets a vertical one, and those two are offset along
# different axes, so there is no one direction to set dots side by side in.
MARKER_SIZE = 70
MARKER_EDGEWIDTH = 0.9


def load_sweep(data_dir, run_name):
	"""Return (latency_ms, horizons, iterations) for one run.

	`latency_ms` is (iterations, horizons), matching the axis order
	`collect_latency_data.py` writes and labelled by the two returned axes.
	"""
	path = data_dir / f"{run_name}.npz"
	if not path.exists():
		raise FileNotFoundError(
			f"missing sweep: {path}\n"
			f"collect it with: python analysis/collect_latency_data.py "
			f"--config-name=<evaluate config> runner.cfg.run_name={run_name} ..."
		)
	with np.load(path) as data:
		return (
			data["plan_latency_means"] * 1e3,
			data["horizons"],
			data["iterations"],
		)


def reachable_frontier(latency, horizons, iterations, budget):
	"""Staircase through the furthest reachable config at each iteration level.

	Returns (line_pts, furthest_pts, clipped_row) in (horizon, iterations)
	coordinates.  The staircase steps left then up between rows, so it passes
	only through reachable configurations.

	A row whose furthest reachable configuration is the last horizon swept has no
	known frontier: the budget was not what stopped it, the grid was, and the real
	frontier is somewhere off to the right (TD-MPC2 at 30 ms reaches H=10 on 5 ms
	of its budget).  Those rows are left out of the line, which instead runs out
	to the last measured configuration and ends there, undotted.  `clipped_row`
	names the highest of them -- the iteration level that happens at -- and is None
	when the budget binds everywhere.  Both point arrays are None when there is
	nothing to draw: no configuration fits the budget, or every one of them does
	and no part of the frontier was pinned down.

	The dot at the top end is dropped on the same grounds: if the highest iteration
	count swept still reaches something, the frontier continues above the grid, and
	only a line that stops lower down has genuinely ended.

	Latency rises with iterations, so the clipped rows are the bottom of the grid
	and the known ones sit above them; a row that broke that order would be
	dropped from the line rather than drawn out of sequence.
	"""
	furthest = []  # (row, col) of max reachable horizon per iteration row
	for r in range(len(iterations)):
		cols = [c for c in range(len(horizons)) if latency[r, c] <= budget]
		if cols:
			furthest.append((r, max(cols)))
	if not furthest:
		return None, None, None

	last_col = len(horizons) - 1
	clipped = [r for r, c in furthest if c == last_col]
	clipped_row = iterations[max(clipped)] if clipped else None
	known = [(r, c) for r, c in furthest if c < last_col and (not clipped or r > max(clipped))]
	if not known:
		return None, None, clipped_row

	r0, c0 = known[0]
	pts = []
	if clipped_row is not None:
		# Out to the last column and no further: the line ends there without a
		# dot, since that configuration is not known to be the frontier.
		pts.append((horizons[last_col], clipped_row))
		pts.append((horizons[c0], clipped_row))        # step left
	pts.append((horizons[c0], iterations[r0]))
	prev_r = r0
	for r, c in known[1:]:
		pts.append((horizons[c], iterations[prev_r]))  # step left
		pts.append((horizons[c], iterations[r]))       # step up
		prev_r = r

	markers = [(horizons[c], iterations[r]) for r, c in known]
	# Same at the top end: a dot there would say the frontier stops, but when the
	# highest iteration count tested still has something reachable it is the sweep
	# that stopped, and the frontier goes on above the plot.  A line that ends
	# lower down does end -- nothing at all is reachable in the row above it -- so
	# that dot stays.
	if known[-1][0] == len(iterations) - 1:
		markers = markers[:-1]
	return np.array(pts), np.array(markers), clipped_row


def unit_segments(points):
	"""A staircase as the unit steps between neighbouring grid points.

	Splitting to unit steps is what turns "do these two methods share this piece
	of line?" into a dictionary lookup: two frontiers overlap exactly where they
	contain the same steps.  Each step is returned as a sorted pair of endpoints,
	so it counts as the same step whichever direction it is walked in.
	"""
	steps = []
	for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
		if x0 == x1:
			stride = 1 if y1 > y0 else -1
			steps += [((x0, y), (x0, y + stride)) for y in range(y0, y1, stride)]
		else:
			stride = 1 if x1 > x0 else -1
			steps += [((x, y0), (x + stride, y0)) for x in range(x0, x1, stride)]
	return [tuple(sorted(step)) for step in steps]


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=DATA_DIR,
		help="directory holding the <run_name>.npz sweeps (default: analysis/planner_latencies)",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=HERE / "reachable_config_contours",
		help="output path without extension; .png and .pdf are both written",
	)
	args = parser.parse_args()

	sweeps = [
		(label, colour, budgets, *load_sweep(args.data_dir, run))
		for label, run, colour, budgets in RUNS
	]

	# Budgets in the order they will be keyed, checked up front rather than at the
	# point of drawing: a run listing a budget with no style is a mistake in RUNS,
	# and it should be reported before any work is done, naming all of them.
	budgets_used = sorted({budget for _, _, budgets, *_ in sweeps for budget in budgets})
	unstyled = [budget for budget in budgets_used if budget not in BUDGET_STYLE]
	if unstyled:
		raise KeyError(
			f"no BUDGET_STYLE entry for budget(s) {unstyled}; add one per budget "
			f"named in RUNS (styles in use: {sorted(BUDGET_STYLE)})"
		)

	# Union of every tested configuration, for the background grid.
	all_horizons = np.unique(np.concatenate([h for *_, h, _ in sweeps]))
	all_iterations = np.unique(np.concatenate([i for *_, i in sweeps]))
	n_horizons = min(N_HORIZONS, int(all_horizons.max()))
	n_iterations = min(N_ITERATIONS, int(all_iterations.max()))

	plt.rcParams.update(
		{
			"font.family": "sans-serif",
			"font.size": 17,
			"axes.spines.top": False,
			"axes.spines.right": False,
			"axes.linewidth": 0.8,
			"axes.grid": True,
			"grid.color": "#e1e0d9",
			"grid.linewidth": 0.8,
			"figure.dpi": 150,
		}
	)

	fig, ax = plt.subplots(figsize=(7.4, 6.4))
	# Square axes: both axes count the same kind of thing (planner steps), and a
	# staircase only reads as one when a step right and a step up are the same
	# size on the page. `set_box_aspect` squares the box itself rather than tying
	# it to the data range, so it holds whatever view window is in force.
	ax.set_box_aspect(1)

	gx, gy = np.meshgrid(all_horizons, all_iterations)
	ax.scatter(gx, gy, s=30, color="#c3c2b7", zorder=1)

	# Work out who draws what before drawing any of it: a segment's width has to
	# be divided by the number of frontiers on it, which is not known until every
	# frontier has been walked. One entry per (method, budget) pair -- that is
	# what a single line on this figure is.
	frontiers = []                    # (colour, linestyle) per drawn frontier
	segment_owners = defaultdict(list)  # unit step -> indices into `frontiers`
	marker_owners = defaultdict(list)   # grid point -> indices into `frontiers`

	for label, colour, budgets, latency, horizons, iterations in sweeps:
		for budget in budgets:
			pts, markers, clipped_row = reachable_frontier(latency, horizons, iterations, budget)
			if pts is None:
				# Opposite situations, and both leave nothing to draw: either no
				# configuration fits the budget, or all of them do and the frontier
				# is somewhere off the grid entirely.
				print(
					f"{label} @{budget:g} ms: nothing reachable"
					if clipped_row is None
					else f"{label} @{budget:g} ms: every tested configuration fits, so "
					f"the whole frontier is past H={int(horizons[-1])} -- nothing to draw"
				)
				continue
			index = len(frontiers)
			frontiers.append((colour, BUDGET_STYLE[budget]))
			for step in unit_segments(pts):
				segment_owners[step].append(index)
			for point in markers:
				marker_owners[(int(point[0]), int(point[1]))].append(index)
			if clipped_row is not None:
				# Said rather than drawn: the line simply stops at the edge of what
				# was measured, which the plot cannot distinguish from any other
				# stretch of line.
				print(
					f"{label} @{budget:g} ms: frontier is past H={int(horizons[-1])} "
					f"at {int(clipped_row)} iterations and below, so the line runs out "
					f"to the last configuration measured and ends there, undotted"
				)

	# Thin the lanes, if they need it, until the most-shared segment in the figure
	# is no wider than the coloured part of a dot -- a stack broader than the
	# configuration it runs through stops reading as a line on the grid.  One
	# scale for the whole figure, so every line keeps the same width whatever it
	# shares.
	widest_stack = max((len(owners) for owners in segment_owners.values()), default=1)
	stack_width = widest_stack * LINEWIDTH + (widest_stack - 1) * LANE_GAP
	scale = min(1.0, (np.sqrt(MARKER_SIZE) - MARKER_EDGEWIDTH) / stack_width)
	width, gap = LINEWIDTH * scale, LANE_GAP * scale

	for (start, end), owners in segment_owners.items():
		# The lanes are laid out symmetrically about the true position, so a
		# segment with one owner is drawn exactly where it would have been
		# anyway, and a shared one straddles it evenly.  The offset is in points
		# rather than data units: it separates lines whose width is measured in
		# points, and should not change with the axis range.
		horizontal = start[1] == end[1]
		for rank, index in enumerate(owners):
			colour, linestyle = frontiers[index]
			offset = (rank - (len(owners) - 1) / 2) * (width + gap)
			transform = offset_copy(
				ax.transData,
				fig=fig,
				x=0 if horizontal else offset,
				y=offset if horizontal else 0,
				units="points",
			)
			ax.plot(
				(start[0], end[0]),
				(start[1], end[1]),
				linestyle=linestyle,
				color=colour,
				linewidth=width,
				# Fills the notch the butt ends would leave where a horizontal
				# sub-band meets a vertical one at a corner of the staircase.
				solid_capstyle="projecting",
				transform=transform,
				zorder=3,
			)

	# Emphasise the furthest reachable configs the lines pass through.  Shared
	# points nest, largest first, so every owner keeps a visible ring.
	for point, owners in marker_owners.items():
		for rank, index in enumerate(owners):
			colour, _ = frontiers[index]
			ax.scatter(
				point[0],
				point[1],
				s=MARKER_SIZE * ((len(owners) - rank) / len(owners)) ** 2,
				color=colour,
				# Only the outermost dot is separated from the line behind it; an
				# edge on the inner ones would cover the ring they sit in.
				edgecolor="white" if rank == 0 else "none",
				linewidth=MARKER_EDGEWIDTH,
				zorder=4 + rank,
			)

	# Zoom to the view window; lines/dots outside it are simply clipped.
	ax.set_xticks(np.arange(1, n_horizons + 1))
	ax.set_yticks(np.arange(1, n_iterations + 1))
	ax.set_xlim(0.3, n_horizons + 0.7)
	ax.set_ylim(0.3, n_iterations + 0.7)
	ax.set_xlabel("Planning Horizon")
	ax.set_ylabel("Plan Optimization Iterations")

	ax.tick_params(labelsize=15, color="#c3c2b7")
	ax.set_axisbelow(True)

	# Before the legends, not after: `tight_layout` resizes the axes, and the
	# legends are positioned in axes coordinates.
	fig.tight_layout()

	# Two legends stacked down the right-hand side: colour = method (top), line
	# style = budget (below it).
	method_handles = [
		Line2D([0], [0], color=colour, linewidth=width, label=label)
		for label, colour, *_ in sweeps
	]
	# Only the budgets some method is actually drawn at: with the budgets set per
	# method, a key listing any others would describe lines that are not there.
	budget_handles = [
		Line2D([0], [0], color="#52514e", linewidth=width,
			   linestyle=BUDGET_STYLE[b], label=f"{b:g} ms")
		for b in budgets_used
	]
	method_legend = ax.legend(
		handles=method_handles,
		title="Method",
		frameon=True,
		facecolor=LEGEND_FACECOLOR,
		edgecolor=LEGEND_EDGECOLOR,
		framealpha=1.0,
		borderpad=0.8,
		loc="upper left",
		bbox_to_anchor=(LEGEND_X, 1.0),
		alignment="center",
		fontsize=LEGEND_FONTSIZE,
		# `title_fontproperties` rather than `title_fontsize`: matplotlib rejects
		# the two together, and the weight has to come from somewhere.
		title_fontproperties={"weight": "bold", "size": LEGEND_FONTSIZE + 1},
	)
	ax.add_artist(method_legend)

	# Hang the budget legend off the bottom of the method legend as it actually
	# rendered, rather than at a guessed offset that would collide the moment
	# RUNS grows or the font changes. Legends outside the axes are invisible to
	# `tight_layout`, so this stays put; `bbox_extra_artists` keeps them in frame
	# when the figure is saved.
	fig.canvas.draw()
	method_box = method_legend.get_window_extent().transformed(ax.transAxes.inverted())
	budget_legend = ax.legend(
		handles=budget_handles,
		title="Planning Budget",
		frameon=True,
		facecolor=LEGEND_FACECOLOR,
		edgecolor=LEGEND_EDGECOLOR,
		framealpha=1.0,
		borderpad=0.8,
		loc="upper left",
		bbox_to_anchor=(LEGEND_X, method_box.y0 - 0.06),
		alignment="center",
		fontsize=LEGEND_FONTSIZE,
		# `title_fontproperties` rather than `title_fontsize`: matplotlib rejects
		# the two together, and the weight has to come from somewhere.
		title_fontproperties={"weight": "bold", "size": LEGEND_FONTSIZE + 1},
	)

	extra = (method_legend, budget_legend)
	for ext in ("png", "pdf"):
		out = args.out.with_suffix(f".{ext}")
		fig.savefig(out, dpi=300, bbox_inches="tight", bbox_extra_artists=extra)
		print(f"wrote {out}")
	plt.close(fig)


if __name__ == "__main__":
	main()
