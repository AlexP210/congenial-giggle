"""Iso-budget planner-config frontiers: Student vs. Teacher.

For a given planning-latency budget, which MPPI configurations (planning horizon
x optimization iterations) can each model actually run in time? Horizon and
iterations are integers and latency increases monotonically with both, so the
set reachable within a budget is a lower-left "staircase" region. Each frontier
is drawn with horizontal/vertical segments through the furthest reachable tested
configuration at every iteration level -- everything below-and-left of a model's
line is reachable within that budget. Because the student plans far faster, its
staircase for a given budget sits much further out than the teacher's.

Grid: horizon on x, iterations on y (high-resolution 20x20 cheetah-jump latency
sweep). Frontiers are computed on the full grid; a configurable view window
(N_HORIZONS / N_ITERATIONS) below zooms the axes to the lower-left corner. In
the raw arrays axis 0 indexes iterations and axis 1 indexes horizon (the column
axis carries the steeper latency slope, matching the earlier real-task sweep).

Run from this directory (or anywhere -- paths are resolved relative to the
script location):

    python trc_reachable_config_contours.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# Filenames for the cheetah-jump latency sweep.
FILES = {"student": "StudentLatencyCheetahJump", "teacher": "TeacherLatencyCheetahJump"}
COLORS = {"student": "#2a78d6", "teacher": "#eb6834"}
LABELS = {
    "student": "Student (distilled from Teacher)",
    "teacher": "Multi-Task TD-MPC2 Teacher (fine-tuned)",
}

# View window. The frontiers and tested-config dots are always computed and
# drawn on the FULL grid; these only zoom the axes to the lower-left corner.
# Set to a value >= the data size to see the whole sweep. Clamped to the data.
N_HORIZONS = 10     # planning-horizon steps to *show* (x-axis limit)
N_ITERATIONS = 10   # optimization-iteration steps to *show* (y-axis limit)

_data_shape = np.load(HERE / f"{FILES['student']}_plan_latency_means.npy").shape
N_ITERATIONS = min(N_ITERATIONS, _data_shape[0])
N_HORIZONS = min(N_HORIZONS, _data_shape[1])

# Full swept grid (raw arrays: rows = iterations, cols = horizon).
HORIZONS = np.arange(1, _data_shape[1] + 1)     # x-axis (full)
ITERATIONS = np.arange(1, _data_shape[0] + 1)   # y-axis (full)

# Planning budgets (ms). Chosen in the range both models span so each budget
# yields a Student *and* a Teacher frontier for a direct comparison.
BUDGETS_MS = [10, 20, 30]
BUDGET_STYLE = {10: ":", 20: "--", 30: "-"}


def load_latency_ms(name):
    """Mean planning latency (ms) over the full sweep, shape (iterations, horizon)."""
    return np.load(HERE / f"{FILES[name]}_plan_latency_means.npy") * 1e3


def reachable_frontier(latency, budget):
    """Staircase through the furthest reachable config at each iteration level.

    Returns (line_pts, furthest_pts) in (horizon, iterations) coordinates, or
    (None, None) if no configuration is reachable. The staircase steps left then
    up between rows, so it passes only through reachable configurations.
    """
    furthest = []  # (row, col) of max reachable horizon per iteration row
    for r in range(len(ITERATIONS)):
        cols = [c for c in range(len(HORIZONS)) if latency[r, c] <= budget]
        if cols:
            furthest.append((r, max(cols)))
    if not furthest:
        return None, None

    r0, c0 = furthest[0]
    pts = [(HORIZONS[c0], ITERATIONS[r0])]
    prev_r = r0
    for r, c in furthest[1:]:
        pts.append((HORIZONS[c], ITERATIONS[prev_r]))  # step left
        pts.append((HORIZONS[c], ITERATIONS[r]))       # step up
        prev_r = r
    markers = [(HORIZONS[c], ITERATIONS[r]) for r, c in furthest]
    return np.array(pts), np.array(markers)


def main():
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

    fig, ax = plt.subplots(figsize=(8.0, 6.8))

    # Faint markers for every tested configuration (shared grid).
    gx, gy = np.meshgrid(HORIZONS, ITERATIONS)
    ax.scatter(gx, gy, s=30, color="#c3c2b7", zorder=1)

    for name in ("student", "teacher"):
        latency = load_latency_ms(name)
        color = COLORS[name]
        for budget in BUDGETS_MS:
            pts, markers = reachable_frontier(latency, budget)
            if pts is None:
                continue
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                linestyle=BUDGET_STYLE[budget],
                color=color,
                linewidth=2.8,
                zorder=3,
            )
            # Emphasise the furthest reachable configs the line passes through.
            ax.scatter(
                markers[:, 0],
                markers[:, 1],
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )

    # Zoom to the view window; lines/dots outside it are simply clipped.
    ax.set_xticks(np.arange(1, N_HORIZONS + 1))
    ax.set_yticks(np.arange(1, N_ITERATIONS + 1))
    ax.set_xlim(0.3, N_HORIZONS + 0.7)
    ax.set_ylim(0.3, N_ITERATIONS + 0.7)
    ax.set_xlabel("Planning horizon")
    ax.set_ylabel("Plan optimization iterations")

    # Two legends stacked above the figure: colour = model (top, one per row so
    # the long labels fit), line style = budget (bottom row).
    model_handles = [
        Line2D([0], [0], color=COLORS[n], linewidth=2.8, label=LABELS[n])
        for n in ("student", "teacher")
    ]
    budget_handles = [
        Line2D([0], [0], color="#52514e", linewidth=2.8,
               linestyle=BUDGET_STYLE[b], label=f"{b} ms")
        for b in BUDGETS_MS
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=1,
    )
    ax.add_artist(model_legend)
    budget_legend = ax.legend(
        handles=budget_handles,
        title="Planning budget",
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(budget_handles),
    )

    ax.tick_params(labelsize=15, color="#c3c2b7")
    ax.set_axisbelow(True)

    fig.tight_layout()
    extra = (model_legend, budget_legend)
    for ext in ("png", "pdf"):
        out = HERE / f"trc_reachable_config_contours.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", bbox_extra_artists=extra)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
