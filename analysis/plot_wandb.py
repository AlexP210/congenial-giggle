"""
Poll wandb runs and generate plots of training metrics.

Usage:
    python plot_wandb.py --entity <entity> --project <project> [options]

Options:
    --entity        wandb entity (username or team)
    --project       wandb project name
    --runs          comma-separated run IDs or names to include (default: all)
    --metric        metric key to plot (default: episode_return)
    --x-axis        x-axis key (default: _step)
    --out           output directory for plots (default: .)
    --smoothing     exponential moving average smoothing factor 0-1 (default: 0.0)
"""

import argparse
import os
from pathlib import Path

import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import wandb


def fetch_runs(entity: str, project: str, run_ids: str | list[str] | None = None) -> list:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    if run_ids is not None:
        ids = [run_ids] if isinstance(run_ids, str) else run_ids
        runs = [r for r in runs if r.id in ids or r.name in ids]
    return list(runs)


def fetch_runs_by_name(entity: str, project: str, name_prefix: str, per_page: int = 200) -> dict[str, object]:
    """Fetch all runs whose name starts with name_prefix."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=per_page)
    return {r.name: r for r in runs if r.name.startswith(name_prefix)}


def smooth(values: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0.0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = factor * smoothed[i - 1] + (1 - factor) * values[i]
    return smoothed


def fetch_history(run, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    df = run.history(keys=[x_key, y_key], pandas=True)
    df = df.dropna(subset=[y_key])
    x = df[x_key].to_numpy() if x_key in df.columns else np.arange(len(df))
    y = df[y_key].to_numpy()
    return x, y


def plot_metric(
    runs: list,
    x_key: str,
    y_key: str,
    smoothing: float,
    out_dir: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for run in runs:
        try:
            x, y = fetch_history(run, x_key, y_key)
        except Exception as e:
            print(f"  Skipping run '{run.name}': {e}")
            continue
        if len(y) == 0:
            print(f"  No data for '{y_key}' in run '{run.name}', skipping.")
            continue
        y_plot = smooth(y, smoothing)
        ax.plot(x, y_plot, label=run.name, alpha=0.85)
        if smoothing > 0.0:
            ax.plot(x, y, alpha=0.2, color=ax.lines[-1].get_color())

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(y_key)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()

    safe_key = y_key.replace("/", "_")
    out_path = out_dir / f"{safe_key}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def fetch_max_metric(run, metric_key: str) -> float:
    """Return the maximum logged value of metric_key for a single run.

    Reads from run.summary first (no extra API call). Falls back to full
    history only if the key is absent from summary.
    """
    val = run.summary.get(metric_key)
    if val is not None:
        return float(val)
    # fallback: download history (slow)
    df = run.history(keys=[metric_key], pandas=True).dropna(subset=[metric_key])
    return float(df[metric_key].max()) if not df.empty else float("nan")


def plot_heatmap(
    runs: dict,
    betas: tuple,
    dims: tuple,
    metric_key: str = "evaluation/real_task/episode_return",
    out_path: str | Path = "heatmap.png",
):
    """Plot a (latent_dim x beta) heatmap of max metric_key values."""
    data = np.full((len(dims), len(betas)), fill_value=float("nan"))
    for i, dim in enumerate(dims):
        for j, beta in enumerate(betas):
            run_list = runs.get((beta, dim), [])
            if not run_list:
                print(f"  No run for beta={beta}, dim={dim}")
                continue
            val = fetch_max_metric(run_list[0], metric_key)
            data[i, j] = val
            print(f"  beta={beta}, dim={dim}: max={val:.2f}")

    fig, ax = plt.subplots(figsize=(max(6, len(betas) * 1.4), max(5, len(dims) * 1.1)))
    sns.heatmap(
        data,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        xticklabels=[str(b) for b in betas],
        yticklabels=[str(d) for d in dims],
        cbar_kws={"label": f"Max {metric_key.split('/')[-1]}"},
    )
    ax.set_xlabel("Beta")
    ax.set_ylabel("Latent Dim")
    ax.set_title(f"Max {metric_key.split('/')[-1]}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_lines_by_dim(
    runs: dict,
    betas: tuple,
    dims: tuple,
    metric_key: str = "evaluation/simulation_task/episode_return",
    x_key: str = "_step",
    out_path: str | Path = "lines_by_dim.png",
):
    """One subplot per latent dim; each subplot has one line per beta value."""
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4), sharey=True)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(betas)))

    for ax, dim in zip(axes, dims):
        for color, beta in zip(colors, betas):
            run_list = runs.get((beta, dim), [])
            if not run_list:
                continue
            x, y = fetch_history(run_list[0], x_key, metric_key)
            ax.plot(x, y, label=f"β={beta}", color=color)
        ax.set_title(f"dim={dim}")
        ax.set_xlabel(x_key)

    axes[0].set_ylabel(metric_key.split("/")[-1])

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


print("Fetching runs (single API call).")
all_runs = fetch_runs_by_name(
    entity="<wandb-entity>",
    project="Squeeze-to-Plan",
    name_prefix="LatentDimBetaTest_",
)
print(f"  Found {len(all_runs)} run(s).")

# Discover betas and dims from the actual run names
betas_set, dims_set = set(), set()
for name in all_runs:
    m = re.match(r"LatentDimBetaTest_BETA([\d.]+)_DIM(\d+)", name)
    if m:
        betas_set.add(float(m.group(1)))
        dims_set.add(int(m.group(2)))
BETAS = tuple(sorted(betas_set))
DIMS = tuple(sorted(dims_set))
print(f"  Betas: {BETAS}")
print(f"  Dims:  {DIMS}")

runs = {}
for name, run in all_runs.items():
    m = re.match(r"LatentDimBetaTest_BETA([\d.]+)_DIM(\d+)", name)
    if m:
        runs[(float(m.group(1)), int(m.group(2)))] = [run]

print("Plotting heatmap.")
plot_heatmap(runs, betas=BETAS, dims=DIMS, out_path="latent_dim_beta_heatmap.png")

