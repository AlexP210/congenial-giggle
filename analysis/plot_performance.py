"""
Fetch wandb runs and plot a bar chart with per-bar data specification.

Usage:
    python plot_performance.py \\
        --bars  <run-id>:<key> [<run-id>:<key> ...] \\
        --errors <run-id>:<key> [<run-id>:<key> ...] \\
        [--labels <label> ...] [--out <path>] [--title <title>]

Each --bars entry selects a single bar: the wandb summary value at <key> from <run-id>.
Each --errors entry selects the corresponding error bar value in the same format.
The number of --bars and --errors entries must be equal.

Example:
    python plot_performance.py \\
        --bars  abc123:eval/sim/episode_return  abc123:eval/real/episode_return \\
        --errors abc123:eval/sim/episode_return_sem abc123:eval/real/episode_return_sem \\
        --labels "Sim task" "Real task"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def parse_specifier(spec: str) -> tuple[str, str]:
    """Split 'run-id:some/summary/key' on the first colon."""
    idx = spec.index(":")
    return spec[:idx], spec[idx + 1:]


def fetch_runs(entity: str, project: str, run_ids: set[str]) -> dict[str, object]:
    api = wandb.Api()
    result: dict[str, object] = {}
    for r in api.runs(f"{entity}/{project}"):
        if r.id in run_ids or r.name in run_ids:
            result[r.id] = r
            result[r.name] = r
    return result


def get_summary_value(run, key: str) -> float:
    val = run.summary.get(key)
    return float("nan") if val is None else float(val)


def plot_performance(
    values: list[float],
    errors: list[float],
    labels: list[str],
    out_path: Path,
    title: str,
):
    n = len(values)
    x = np.arange(n)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2 + 2), 5))
    ax.bar(
        x, values, width=0.6,
        yerr=errors, capsize=4,
        color=[colors[i % len(colors)] for i in range(n)],
        error_kw={"elinewidth": 1.5},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Episode Return")
    ax.set_title(title)
    plt.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default="<wandb-entity>")
    parser.add_argument("--project", default="Squeeze-to-Plan")
    parser.add_argument("--bars", nargs="+", required=True, metavar="RUN:KEY",
                        help="One <run-id>:<summary-key> per bar")
    parser.add_argument("--errors", nargs="+", required=True, metavar="RUN:KEY",
                        help="One <run-id>:<summary-key> per error bar (must match --bars count)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display label for each bar (must match --bars count)")
    parser.add_argument("--out", default="performance.png")
    parser.add_argument("--title", default="Episode Return")
    args = parser.parse_args()

    if len(args.bars) != len(args.errors):
        raise ValueError(f"--bars ({len(args.bars)}) and --errors ({len(args.errors)}) must have equal counts")
    if args.labels and len(args.labels) != len(args.bars):
        raise ValueError(f"--labels ({len(args.labels)}) must match --bars count ({len(args.bars)})")

    bar_specs = [parse_specifier(s) for s in args.bars]
    err_specs = [parse_specifier(s) for s in args.errors]

    needed_ids = {run_id for run_id, _ in bar_specs + err_specs}

    print("Fetching runs...")
    run_map = fetch_runs(args.entity, args.project, needed_ids)
    missing = needed_ids - set(run_map.keys())
    if missing:
        print(f"Warning: could not find runs: {missing}")

    values, errors = [], []
    for (bar_run, bar_key), (err_run, err_key) in zip(bar_specs, err_specs):
        run = run_map.get(bar_run)
        values.append(float("nan") if run is None else get_summary_value(run, bar_key))
        erun = run_map.get(err_run)
        errors.append(float("nan") if erun is None else get_summary_value(erun, err_key))

    labels = args.labels if args.labels else [f"{rid}:{key}" for rid, key in bar_specs]

    plot_performance(values, errors, labels, Path(args.out), args.title)


if __name__ == "__main__":
    main()
