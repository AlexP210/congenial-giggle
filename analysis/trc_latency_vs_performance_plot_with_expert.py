"""Plot episode return vs. planning-latency budget for the MPPI planner.

Compares the "student" and "teacher" models against a single-task "expert" on
the cheetah-flip task. Each model was evaluated over a 4x4 grid of MPPI
configurations; every cell yields a mean planning latency (seconds) and a mean
episode return, each with an associated standard error of the mean (SEM). We
flatten the grid and plot return against the measured planning budget (ms), so
the picture is performance vs. compute.

Run from this directory (or anywhere -- paths are resolved relative to the
script location):

    python trc_latency_vs_performance_plot_with_expert.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# CVD-safe categorical trio (blue / orange / green).
COLORS = {"student": "#2a78d6", "teacher": "#eb6834", "expert": "#2ca02c"}
LABELS = {
    "teacher": "Multi-Task TD-MPC2 Teacher (fine-tuned)",
    "student": "Student (distilled from Teacher)",
    "expert": "Single-Task TD-MPC2 (from scratch)",
}

# File-name prefix for each model's cheetah-flip evaluation .npy files.
PREFIXES = {
    "student": "student_cheetah_flip",
    "teacher": "teacher_cheetah_flip_finetuned",
    "expert": "expert_cheetah_flip",
}

# Left edge of the (log) budget axis. A log axis cannot render x = 0, so the
# frontier rises from zero return at this edge -- the visual of the implied
# origin: with no planning budget there is no performance.
X_LEFT_MS = 1.0


def load_prefix(prefix):
    """Return flattened (latency_ms, latency_sem_ms, return, return_sem) arrays.

    File names are built as ``{prefix}_{kind}.npy``.
    """

    def load(kind):
        return np.load(HERE / f"{prefix}_{kind}.npy").ravel()

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


def plot_model(ax, name, lat, lat_sem, ret, ret_sem):
    """Draw one model's scatter and best-within-budget frontier."""
    color = COLORS[name]

    # All configurations with x/y error bars.
    ax.errorbar(
        lat,
        ret,
        xerr=lat_sem,
        yerr=ret_sem,
        fmt="o",
        markersize=7,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        ecolor=color,
        elinewidth=1.3,
        capsize=3.0,
        alpha=0.85,
        color=color,
        label=LABELS[name],
        zorder=3,
    )

    # Best-return-within-budget frontier (monotone step). It rises from the
    # implied origin (zero return until the cheapest config's budget) and
    # extends flat to the widest budget so the models are comparable.
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
            linewidth=2.2,
            alpha=0.9,
            zorder=2,
        )
        # Vertical riser at step_x[i + 1] (dotted).
        ax.plot(
            [step_x[i + 1], step_x[i + 1]],
            [step_y[i], step_y[i + 1]],
            linestyle=":",
            color=color,
            linewidth=2.2,
            alpha=0.9,
            zorder=2,
        )


def make_plot(out_basename):
    """Build the return-vs-budget figure comparing student, teacher, expert."""
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for name in ("student", "teacher", "expert"):
        lat, lat_sem, ret, ret_sem = load_prefix(PREFIXES[name])
        plot_model(ax, name, lat, lat_sem, ret, ret_sem)

    ax.set_xscale("log")
    ax.set_xlim(left=X_LEFT_MS)
    ax.set_ylim(bottom=100)
    ax.set_xlabel("Planning budget (ms)")
    ax.set_ylabel("Episode return")

    # Legend: the three model scatters plus one entry explaining the step lines.
    handles, labels = ax.get_legend_handles_labels()
    frontier_proxy = Line2D([0], [0], color="#52514e", linewidth=2.2)
    legend = ax.legend(
        handles + [frontier_proxy],
        labels + ["Best performance within budget"],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    ax.tick_params(labelsize=15, color="#c3c2b7")
    ax.set_axisbelow(True)

    fig.tight_layout()

    # Widen the figure so the axes are at least as wide as the 2x2 legend above
    # them. The horizontal margins (ylabel on the left, padding on the right) are
    # roughly fixed in inches, so grow the figure by the axes/legend width deficit.
    fig.canvas.draw()
    dpi = fig.dpi
    leg_w = legend.get_window_extent().width / dpi
    ax_w = ax.get_window_extent().width / dpi
    if leg_w > ax_w:
        margins = fig.get_size_inches()[0] - ax_w
        fig.set_size_inches(leg_w + margins, fig.get_size_inches()[1])
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

    make_plot("trc_cheetah_flip_latency_vs_performance_with_expert")


if __name__ == "__main__":
    main()
