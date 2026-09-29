"""Heatmap of the Teacher - Student planning-latency gap per planner config.

For every MPPI configuration in the swept grid (planning horizon x optimization
iterations) we take the real-task mean planning latency of each model and plot
the difference (Teacher minus Student) in milliseconds. The teacher is slower in
every configuration, so the difference is all-positive -- a single-hue
sequential map (magnitude), annotated with the value in each cell.

Grid axes match the companion PlanLatency_*_real_task figures: horizon on x,
iterations on y, both swept over [1, 5, 9, 13]. In the raw (4x4) arrays axis 0
indexes iterations and axis 1 indexes horizon.

Run from this directory (or anywhere -- paths are resolved relative to the
script location):

    python trc_teacher_student_latency_diff_heatmap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent

# Swept planner settings (raw arrays: rows = iterations, cols = horizon).
HORIZONS = [1, 5, 9, 13]      # x-axis (columns)
ITERATIONS = [1, 5, 9, 13]    # y-axis (rows)

# Single-hue sequential blue ramp (light -> dark), from the shared palette.
BLUE_RAMP = ["#eaf2fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("trc_blue", BLUE_RAMP)


def load_latency_ms(name):
    """Real-task mean planning latency (ms) for a model, shape (iters, horizon)."""
    return np.load(HERE / f"{name}_real_task_plan_latency_means.npy") * 1e3


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "figure.dpi": 150,
        }
    )

    diff = load_latency_ms("teacher") - load_latency_ms("student")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(diff, cmap=CMAP, origin="lower", aspect="auto")

    # Annotate each cell; text colour flips to stay legible on dark cells.
    thresh = diff.min() + 0.6 * (diff.max() - diff.min())
    for r in range(diff.shape[0]):
        for c in range(diff.shape[1]):
            ax.text(
                c,
                r,
                f"{diff[r, c]:.0f}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if diff[r, c] > thresh else "#0b0b0b",
            )

    ax.set_xticks(range(len(HORIZONS)), HORIZONS)
    ax.set_yticks(range(len(ITERATIONS)), ITERATIONS)
    ax.set_xlabel("Planning horizon")
    ax.set_ylabel("Plan optimization iterations")
    ax.set_title("Teacher − Student planning latency")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Latency gap (ms)")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = HERE / f"trc_teacher_student_latency_diff_heatmap.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
