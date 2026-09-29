import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import seaborn as sns

from tqdm import tqdm

from s2p.models.agent_model import AgentModel
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.evaluators.flops_evaluator import FLOPSEvaluator
from s2p.evaluators.plan_latency_evaluator import PlanLatencyEvaluator

from s2p import PROJECT_ROOT

@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="offline_training")
def main(cfg: DictConfig):

    # Grab the task
    task:OnlineTaskBase = instantiate(cfg.task)
    sample_obs = torch.zeros(size=(1, *task.observation_dimension), device=cfg.device)

    # Grab the plan latency and the flops evaluators
    cfg.evaluators.plan_latency.cfg.num_plans = 5
    latency_evaluator:PlanLatencyEvaluator = instantiate(cfg.evaluators.plan_latency)
    flops_evaluator:FLOPSEvaluator = instantiate(cfg.evaluators.flops)

    horizons_to_check = list(range(1, 21, 2))
    iterations_to_check = list(range(1, 21, 2))
    latent_dimensions_to_check = [2**i for i in range(0, 11, 2)]

    def run_sweep(sweep_values, set_fn, label):
        flops, times, times_sems = [], [], []
        for v in tqdm(sweep_values, desc=label):
            set_fn(v)
            agent: AgentModel = instantiate(cfg.agent)
            flops_info = flops_evaluator(agent)
            flops.append(
                flops_info["encoder/flops"]
                + cfg.planner.cfg.iterations * (
                    cfg.planner.cfg.horizon * (flops_info["dynamics/flops"] + flops_info["reward/flops"])
                    + flops_info["value/flops"]
                )
            )
            latency_info = latency_evaluator(agent)
            times.append(latency_info["plan_latency"])
            times_sems.append(latency_info["plan_latency_sem"])
        return np.array(flops), np.array(times), np.array(times_sems)

    sweeps = [
        (horizons_to_check,          lambda v: setattr(cfg.planner.cfg, "horizon", v),     "Horizon sweep"),
        (iterations_to_check,        lambda v: setattr(cfg.planner.cfg, "iterations", v),  "Iterations sweep"),
        (latent_dimensions_to_check, lambda v: setattr(cfg, "latent_dimension", v),         "Latent dim sweep"),
    ]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["#3d85c8", "#e06c37", "#43a87a"]

    for ax, (values, set_fn, label), color in zip(axes, sweeps, colors):
        flops, times, times_sems = run_sweep(values, set_fn, label)

        ax.scatter(flops, times, color=color, s=50, alpha=0.8, linewidths=0, zorder=3)
        ax.errorbar(
            flops, times,
            yerr=times_sems,
            fmt="none",
            ecolor="gray",
            elinewidth=0.8,
            capsize=2,
            alpha=0.5,
            zorder=2,
        )
        ax.set_xlabel("FLOPs", labelpad=8)
        ax.set_ylabel("Planning latency (s)", labelpad=8)
        ax.set_title(label, pad=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    sns.despine()
    fig.suptitle("Planning latency vs. FLOPs", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig("flops_vs_latency.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
	main()
