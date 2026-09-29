import torch
import os
from omegaconf import OmegaConf

from s2p.models.agent_model import AgentModel
from s2p.models.tdmpc2_world_model import TDMPC2WorldModel
from s2p.models.teacher_student_encoder_model import TeacherStudentEncoderModel
from s2p.evaluators.plan_latency_evaluator import PlanLatencyEvaluator
from s2p.evaluators.real_task_evaluator import RealTaskEvaluator

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import seaborn as sns

from tqdm import tqdm

from s2p.runners.runner_base import RunnerBase
from s2p import PROJECT_ROOT

@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	
	runner:RunnerBase = instantiate(cfg.runner)
	model = runner.model
	
	# Load the model
	model_name = runner.cfg.run_name

	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	# Plan latency
	plan_latency_evaluator:PlanLatencyEvaluator = instantiate(cfg.evaluators.plan_latency)
	plan_latency_evaluator.cfg.num_plans = 10

	# Task
	task_evaluator:RealTaskEvaluator = instantiate(cfg.evaluators.real_task_planning)
	task_evaluator.cfg.num_episodes = 50

	# The horizons and iteration counts to test
	horizons = range(1, 13)
	iterations = range(1, 13)
	
	# Store the results
	plan_latency_means = np.empty(shape=(len(horizons), len(iterations)))
	plan_latency_sem = np.empty(shape=(len(horizons), len(iterations)))
	return_means = np.empty(shape=(len(horizons), len(iterations)))
	return_sem = np.empty(shape=(len(horizons), len(iterations)))

	# Call evaluators once to take care of any first-time set-up
	_ = plan_latency_evaluator(model)

	# Run the test
	pbar = tqdm(total=len(horizons) * len(iterations))
	for h_idx, h in enumerate(horizons):
		for i_idx, i in enumerate(iterations):

			print(f"Horizon={h}, Iterations={i}")

			model.planner.cfg.horizon = h
			model.planner.cfg.iterations = i
			info = plan_latency_evaluator(model)
			plan_latency_means[h_idx, i_idx] = info["plan_latency"]
			plan_latency_sem[h_idx, i_idx] = info["plan_latency_sem"]

			info = task_evaluator(model)
			return_means[h_idx, i_idx] = info["episode_success_at_end_rate"]
			return_sem[h_idx, i_idx] = info["episode_success_at_end_rate_sem"]

			pbar.update(1)

	np.save(f"{model_name}_return_means.npy", return_means)
	np.save(f"{model_name}_return_sem.npy", return_sem)
	np.save(f"{model_name}_plan_latency_means.npy", plan_latency_means)
	np.save(f"{model_name}_plan_latency_sem.npy", plan_latency_sem)

	plt.figure(figsize=(6, 5))
	ax = sns.heatmap(
		plan_latency_means*1000,
		cmap="viridis",
		annot=True,
		fmt=".000f",
		xticklabels=horizons,
		yticklabels=iterations,
		cbar_kws={"label": "Time to Plan (ms)"}
	)
	ax.invert_yaxis()
	ax.set_xlabel("Planning Horizon")
	ax.set_ylabel("Plan Optimization Iterations")
	ax.set_title("Time Required for Planning")
	plt.tight_layout()
	plt.savefig(f"PlanLatency_{model_name}.png", dpi=600)

	plt.close()
	plt.clf()

	plt.figure(figsize=(6, 5))
	ax = sns.heatmap(
		return_means,
		cmap="viridis",
		annot=True,
		# fmt=".000f",
		xticklabels=horizons,
		yticklabels=iterations,
		cbar_kws={"label": "Episode Return"}
	)
	ax.invert_yaxis()
	ax.set_xlabel("Planning Horizon")
	ax.set_ylabel("Plan Optimization Iterations")
	ax.set_title("Planner Performance")
	plt.tight_layout()
	plt.savefig(f"PlanPerformance_{model_name}.png", dpi=600)

	plt.close()
	plt.clf()


if __name__ == "__main__":
	main()