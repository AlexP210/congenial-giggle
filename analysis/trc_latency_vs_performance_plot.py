"""Plot episode return vs. planning-latency budget for the MPPI planner.

Compares the "student" and "teacher" models. Each model was evaluated over a
4x4 grid of MPPI configurations; every cell yields a mean planning latency
(seconds) and a mean episode return, each with an associated standard error of
the mean (SEM). We flatten the grid and plot return against the measured
planning budget (ms), so the picture is performance vs. compute.

Two figures are produced, one per evaluation regime:
  * trc_realtime_latency_vs_performance    -- the real-task ("*_real_task_*") data
  * trc_nonrealtime_latency_vs_performance -- the non-realtime data (no infix)

Run from this directory (or anywhere -- paths are resolved relative to the
script location):

    python trc_latency_vs_performance_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# CVD-safe categorical pair (blue / orange).
COLORS = {"student": "#2a78d6", "teacher": "#eb6834"}
LABELS = {"student": "Student", "teacher": "Teacher"}

# Left edge of the (log) budget axis. A log axis cannot render x = 0, so the
# frontier rises from zero return at this edge -- the visual of the implied
# origin: with no planning budget there is no performance.
X_LEFT_MS = 1.0


def load_model(name, infix):
    """Return flattened (latency_ms, latency_sem_ms, return, return_sem) arrays.

    `infix` selects the evaluation regime: "_real_task_" for the real-task data
    or "_" for the non-realtime data.
    """

    def load(kind):
        return np.load(HERE / f"{name}{infix}{kind}.npy").ravel()

    lat_ms = load("plan_latency_means") * 1e3
    lat_sem_ms = load("plan_latency_sem") * 1e3
    ret = load("return_means")
    ret_sem = load("return_sem")
    return lat_ms, lat_sem_ms, ret, ret_sem


def budget_frontier(x, y):
    """Best return achievable within a given latency budget.

    At each budget b, the frontier is max(return) over all configs whose
    latency <= b. This is the running maximum of the latency-sorted returns,
    so it is monotonically non-decreasing -- a proper step curve for every
    model (unlike an upper-left envelope, which can collapse to a single
    point when a model peaks at its cheapest config).
    """
    order = np.argsort(x)
    xs = x[order]
    ys = np.maximum.accumulate(y[order])
    return xs, ys


def make_plot(infix, out_basename):
    """Build one return-vs-budget figure for the given evaluation regime."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for name in ("student", "teacher"):
        lat, lat_sem, ret, ret_sem = load_model(name, infix)
        color = COLORS[name]

        # All configurations with x/y error bars.
        ax.errorbar(
            lat,
            ret,
            xerr=lat_sem,
            yerr=ret_sem,
            fmt="o",
            markersize=9,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.5,
            alpha=0.85,
            color=color,
            label=LABELS[name],
            zorder=3,
        )

        # Best-return-within-budget frontier (monotone step). It rises from the
        # implied origin (zero return until the cheapest config's budget) and
        # extends flat to the widest budget so the two models are comparable.
        # Horizontal treads are solid; the vertical risers are dotted.
        fx, fy = budget_frontier(lat, ret)
        step_x = np.concatenate([[X_LEFT_MS], fx, [lat.max()]])
        step_y = np.concatenate([[0.0], fy, [fy[-1]]])
        for i in range(len(step_x) - 1):
            # Horizontal tread at height step_y[i].
            ax.plot(
                [step_x[i], step_x[i + 1]],
                [step_y[i], step_y[i]],
                linestyle="-",
                color=color,
                linewidth=2.8,
                alpha=0.9,
                zorder=2,
            )
            # Vertical riser at step_x[i + 1] (dotted).
            ax.plot(
                [step_x[i + 1], step_x[i + 1]],
                [step_y[i], step_y[i + 1]],
                linestyle=":",
                color=color,
                linewidth=2.8,
                alpha=0.9,
                zorder=2,
            )

    ax.set_xscale("log")
    ax.set_xlim(left=X_LEFT_MS)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Planning budget (ms)")
    ax.set_ylabel("Episode return")

    # Legend: the two model scatters plus one entry explaining the step lines.
    handles, labels = ax.get_legend_handles_labels()
    frontier_proxy = Line2D([0], [0], color="#52514e", linewidth=2.8)
    ax.legend(
        handles + [frontier_proxy],
        labels + ["Best performance within budget"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    ax.tick_params(labelsize=13, color="#c3c2b7")
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = HERE / f"{out_basename}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.8,
            "figure.dpi": 150,
        }
    )

    make_plot("_real_task_", "trc_realtime_latency_vs_performance")
    make_plot("_", "trc_nonrealtime_latency_vs_performance")


if __name__ == "__main__":
    main()
