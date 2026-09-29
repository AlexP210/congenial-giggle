"""Stage 2 of the evaluation suite: wall-clock plan latency of the planner configuration
in the Hydra config -- no sweep, one configuration per invocation.

Driven by the same top-level Hydra configs as `analysis.py` (any `evaluate_*.yaml`), and
named by the same `runner.cfg.run_name`.

What is timed is one encode followed by one plan, from a fresh episode -- the whole of
what a controller pays between an observation arriving and having an action to execute --
repeated `evaluators.plan_latency.cfg.num_plans` times. Stage 3 (`analyze_success.py`)
reads these numbers back and converts them into the number of env steps a plan costs, so
it never has to re-measure timing inside a task rollout.

Only the parts needed for timing are built: the agent (`cfg.agent`, which carries the
planner and loads its checkpoint) and the plan-latency evaluator
(`cfg.evaluators.plan_latency`). The runner is never instantiated -- it would also build
the task evaluators, and with them a second env and the offline dataset -- but
`cfg.runner.cfg.run_name` is still what names the output entry, so the same `run_name=`
override used elsewhere identifies the run here.

These are latency measurements, so they are only comparable to each other if they share a
device and that device is otherwise idle: one GPU, one run at a time.

The measured list of per-plan latencies is appended, together with the GPU it ran on, to
this run's entry in `analysis/results/<run_name>.json` under the planner's own
`(num_samples, num_elites, horizon, iterations)` -- see `results_store.py`.

Example:

    python analysis/analyze_latency.py \
        --config-name=evaluate_s2p_push_cube \
        runner.cfg.run_name=Analyze_S2P_PushCube \
        hydra.run.dir=/path/to/outputs/hydra \
        data_dir=/path/to/datasets \
        checkpoint_dir=/path/to/pretrained_checkpoints \
        project_root=/path/to/project \
        output_dir=/path/to/outputs \
        device=cuda:0

`num_plans` is whatever the evaluator config says (10 by default) and is not overridden
here, so it can be raised from the command line with
`evaluators.plan_latency.cfg.num_plans=...`. The planner configuration is whatever
`cfg.agent.planner` says -- override `agent.planner.cfg.horizon=...` etc. from the command
line to test a different one.

This supersedes `collect_latency_data.py`, which writes a whole grid to
`analysis/planner_latencies/<run_name>.npz`; that file is kept only because
`plot_reachability.py` still reads its output directory.
"""

import os
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from s2p import PROJECT_ROOT
from s2p.evaluators.plan_latency_evaluator import PlanLatencyEvaluator
from s2p.models.agent_model import AgentModel

import results_store

# Matches `s2p/main.py`, which is the path these models are actually run under; TF32
# matmuls change the latency being measured, so the setting has to be the one deployment
# uses rather than PyTorch's default.
MATMUL_PRECISION = "high"


def _sync(device):
	"""Drain the CUDA queue so work is charged to the config that issued it."""
	if torch.device(device).type == "cuda":
		torch.cuda.synchronize(device)


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	torch.set_float32_matmul_precision(MATMUL_PRECISION)

	# Seeding as `RunnerBase` does it: the runner is skipped here, but the reset states the
	# planner is timed from should still be reproducible.
	torch.manual_seed(cfg.seed)
	torch.cuda.manual_seed_all(cfg.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	run_name = cfg.runner.cfg.run_name

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	plan_latency_evaluator: PlanLatencyEvaluator = instantiate(cfg.evaluators.plan_latency)
	num_plans = plan_latency_evaluator.cfg.num_plans

	# One untimed call first: the first plan pays for CUDA context creation, lazy kernel
	# loads and allocator growth, none of which should be attributed to the measurement.
	_ = plan_latency_evaluator(model)
	_sync(cfg.device)

	planner_cfg = model.planner.cfg
	print(
		f"Timing {results_store.planner_key(planner_cfg)} ({num_plans} plans) for "
		f"{run_name} on {cfg.device}"
	)

	info = plan_latency_evaluator(model, verbose=False)
	_sync(cfg.device)

	entry = {
		"gpu": results_store.gpu_name(cfg.device),
		"latencies": [float(t) for t in info["plan_latencies"]],
	}
	output_path = results_store.add_entry(run_name, planner_cfg, "latency", entry)
	print(f"mean {info['plan_latency'] * 1e3:.1f} ms, appended to {output_path}")


if __name__ == "__main__":
	main()
