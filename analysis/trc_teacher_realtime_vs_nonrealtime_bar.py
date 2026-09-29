"""Bar chart: Teacher's best real-time vs. best non-real-time performance.

For each evaluation regime we take the MPPI configuration achieving the highest
mean episode return and report that return with its standard error of the mean
(SEM), drawn as an error bar. Colours reuse the CVD-safe palette shared with the
latency-vs-performance figures.

Run from this directory (or anywhere -- paths are resolved relative to the
script location):

    python trc_teacher_realtime_vs_nonrealtime_bar.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

MODEL = "teacher"

# CVD-safe categorical pair (blue / orange) -- same scheme as the other figures.
REGIMES = [
    {"label": "Without\nLatency Effects", "infix": "_", "color": "#2a78d6"},
    {"label": "With\nLatency Effects", "infix": "_real_task_", "color": "#eb6834"},
]


def best_return(infix):
    """Return (mean, sem) of the highest-return config for this regime."""
    ret = np.load(HERE / f"{MODEL}{infix}return_means.npy").ravel()
    sem = np.load(HERE / f"{MODEL}{infix}return_sem.npy").ravel()
    i = int(np.argmax(ret))
    return ret[i], sem[i]


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.8,
            "figure.dpi": 150,
        }
    )

    labels = [r["label"] for r in REGIMES]
    colors = [r["color"] for r in REGIMES]
    means, sems = np.array([best_return(r["infix"]) for r in REGIMES]).T

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    x = np.arange(len(REGIMES))

    ax.bar(
        x,
        means,
        width=0.6,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=sems,
        fmt="none",
        ecolor="#0b0b0b",
        elinewidth=1.6,
        capsize=6,
        capthick=1.6,
        zorder=3,
    )

    # Value labels above each error bar.
    for xi, m, s in zip(x, means, sems):
        ax.text(xi, m + s + 0.02 * means.max(), f"{m:.0f}",
                ha="center", va="bottom", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best episode return")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.12)
    ax.tick_params(labelsize=13, color="#c3c2b7")
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = HERE / f"trc_teacher_realtime_vs_nonrealtime_bar.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
