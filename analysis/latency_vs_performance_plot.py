"""Plot episode return vs. planning-latency budget for the MPPI planner.

Same picture as ``trc_latency_vs_performance_plot.py`` -- performance vs.
compute -- but without the best-within-budget frontier, and for whichever runs
you name on the command line. Each run was evaluated over a grid of MPPI
configurations; every cell yields a mean planning latency (seconds) and a mean
episode return, each with an associated standard error of the mean (SEM). We
flatten the grid and plot return against the measured planning budget (ms).

A run named ``foo`` is read from ``foo_plan_latency_means.npy``,
``foo_plan_latency_sem.npy``, ``foo_return_means.npy`` and
``foo_return_sem.npy`` in the data directory (the script's own directory by
default).

    python latency_vs_performance_plot.py student teacher
    python latency_vs_performance_plot.py student_real_task teacher_real_task \
        -o realtime_latency_vs_performance
    python latency_vs_performance_plot.py \
        "student_cheetah_flip=Student" "expert_cheetah_flip=Single-task expert"
    python latency_vs_performance_plot.py --list
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# CVD-safe categorical palette (blue / orange / green / purple / brown / pink).
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

# The four .npy suffixes that make up one run.
KINDS = ("plan_latency_means", "plan_latency_sem", "return_means", "return_sem")


def load_run(data_dir, name):
    """Return flattened (latency_ms, latency_sem_ms, return, return_sem) arrays."""

    def load(kind):
        return np.load(data_dir / f"{name}_{kind}.npy").ravel()

    lat_ms = load("plan_latency_means") * 1e3
    lat_sem_ms = load("plan_latency_sem") * 1e3
    ret = load("return_means")
    ret_sem = load("return_sem")
    return lat_ms, lat_sem_ms, ret, ret_sem


def available_runs(data_dir):
    """Names in `data_dir` that have all four .npy files present."""
    stems = {p.stem for p in data_dir.glob("*.npy")}
    names = {
        stem[: -len(kind) - 1]
        for stem in stems
        for kind in KINDS
        if stem.endswith(f"_{kind}")
    }
    return sorted(n for n in names if all(f"{n}_{k}" in stems for k in KINDS))


def parse_run_spec(spec):
    """Split a ``name`` or ``name=Legend label`` argument into (name, label)."""
    name, sep, label = spec.partition("=")
    if sep and label:
        return name, label
    return name, name.replace("_", " ").title()


def make_plot(runs, data_dir, out_path, xlim_left, ylim_bottom):
    """Build one return-vs-budget figure for the given runs."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for i, (name, label) in enumerate(runs):
        lat, lat_sem, ret, ret_sem = load_run(data_dir, name)
        color = PALETTE[i % len(PALETTE)]

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
            label=label,
            zorder=3,
        )

    ax.set_xscale("log")
    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)
    ax.set_ylim(bottom=ylim_bottom)
    ax.set_xlabel("Planning budget (ms)")
    ax.set_ylabel("Episode return")

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(runs), 3),
        columnspacing=1.5,
        handletextpad=0.5,
    )
    ax.tick_params(labelsize=13, color="#c3c2b7")
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_path.parent / f"{out_path.name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs",
        nargs="*",
        metavar="RUN",
        help=(
            "Run name, i.e. the .npy file-name prefix (e.g. 'student'). "
            "Use 'RUN=Label' to override the legend label."
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="BASENAME",
        help="Output basename, optionally with a directory (.png and .pdf are "
        "written). A bare name lands in the data directory. "
        "Default: the run names joined by '_vs_'.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=HERE,
        help="Directory holding the .npy files and receiving the figures "
        "(default: the script's directory).",
    )
    parser.add_argument(
        "--xlim-left",
        type=float,
        default=None,
        help="Left edge of the (log) budget axis in ms (default: auto).",
    )
    parser.add_argument(
        "--ylim-bottom",
        type=float,
        default=0.0,
        help="Bottom of the return axis (default: 0).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the run names available in the data directory and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()

    if args.list:
        for name in available_runs(data_dir):
            print(name)
        return

    if not args.runs:
        sys.exit("no runs given; pass one or more run names (--list shows them)")

    runs = [parse_run_spec(spec) for spec in args.runs]
    missing = [
        f"{name}_{kind}.npy"
        for name, _ in runs
        for kind in KINDS
        if not (data_dir / f"{name}_{kind}.npy").exists()
    ]
    if missing:
        found = available_runs(data_dir)
        sys.exit(
            "missing .npy files in {}:\n  {}\navailable runs:\n  {}".format(
                data_dir, "\n  ".join(missing), "\n  ".join(found) or "(none)"
            )
        )

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

    out_path = Path(args.out or "_vs_".join(name for name, _ in runs))
    if out_path.parent == Path("."):
        out_path = data_dir / out_path
    make_plot(runs, data_dir, out_path, args.xlim_left, args.ylim_bottom)


if __name__ == "__main__":
    main()
