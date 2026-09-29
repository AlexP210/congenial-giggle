"""What a `real_task_planning` eval costs besides planning: the per-env-step wall-clock of
`RealTaskEvaluator`'s loop, and (optionally) plan latency at the eval's `num_envs`.

Driven by the same top-level Hydra configs as `analysis.py` (any `evaluate_*.yaml`), and
named by the same `runner.cfg.run_name`. Feeds `feasible_planner_configs.py`'s eval-time
estimate, which from the L40S latencies alone only knows the planning part.

Step cost
---------
The real-task evaluator (`cfg.evaluators.real_task_planning`) is instantiated as-is, so the
env it times -- `num_envs`, `max_episode_steps`, `save_video`, the task's `sim_backend` and
cameras -- is exactly the one that eval would run. Its episode loop is then replayed with the
plan call removed; each env step does what `RealTaskEvaluator.__call__` does on every step:

    env.step(action)                       physics + rendering the camera observations
    obs.to(device), reward/success bookkeeping
    env.render()                           only when `save_video` is on, as in the eval
    model.encoder_model.encode(obs, state, action)   every step, not only on replans

timed from before `env.step` to after the encoder update, CUDA-synchronised. Actions are
zeros (the `wait` interim action), so contact-heavy physics a real policy would cause is not
exercised; that is expected to be a small effect next to rendering and the encoder. Episode
resets (reset + first encode) are timed separately, once per batch.

One untimed warm-up episode runs first; `+timed_episodes=<n>` more are recorded. The entry
-- every step time and reset time, plus the settings above and the GPU -- is appended to
`analysis/results/step_cost/<run_name>.json` (see `results_store.py`).

Batched plan latency
--------------------
`PlanLatencyEvaluator` times plans at one env. With `num_envs` > 1 the eval plans for every
env at once -- `MPPIPlanner` rolls out `num_envs * num_samples` trajectories in one batch --
which the single-env latency understates. Passing the same grid as
`analyze_latency_flops_grid.py` (`+horizon_values`, `+iterations_values`,
`+num_samples_values`, `+num_elites_values`, the last two paired by index) together with
`+batched_latency_repeats=<n>` also times `model.plan` at the evaluator's `num_envs` for every
cell: a `(1, num_envs, ...)` random observation encoded and planned from, with a
`[horizon, num_envs, action_dim]` prior, exactly the shapes `RealTaskEvaluator` passes. One
untimed warm-up plan, then `n` timed ones, recorded with their peak memory under the
`batched_latency` stage of `analysis/results/<run_name>.json`. A cell that runs out of GPU
memory is recorded as `{"oom": true}` rather than aborting the sweep -- that is itself the
answer for that config at this `num_envs`. Skipped when `num_envs` is 1, where it would only
repeat `latency`.

Seeding, matmul precision and model setup follow `analyze_latency_flops_grid.py`. ManiSkill
with `num_envs` > 1 needs `task.cfg.sim_backend=physx_cuda:<index>` spelled out (PhysX is
process-wide; see the true-repro parallel-envs notes).

Example -- step cost at 10 envs, 200-step episodes, plus batched latency over the grid:

    python analysis/analyze_step_cost.py \\
        --config-name=evaluate_dino_wm_push_cube \\
        runner.cfg.run_name=dino_wm_push_cube \\
        evaluators.real_task_planning.cfg.num_envs=10 \\
        evaluators.real_task_planning.cfg.max_episode_steps=200 \\
        task.cfg.sim_backend=physx_cuda:0 \\
        device=cuda:0 \\
        +timed_episodes=3 \\
        +horizon_values=[1,2,3,4,5] +iterations_values=[1,2,3,4,5] \\
        +num_samples_values=[64,128,256] +num_elites_values=[8,16,32] \\
        +batched_latency_repeats=5 \\
        data_dir=... checkpoint_dir=... project_root=... output_dir=...
"""

import itertools
import os
import random
import time

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from s2p import PROJECT_ROOT
from s2p.evaluators.real_task_evaluator import RealTaskEvaluator
from s2p.lib.utils import episode_boundary
from s2p.models.agent_model import AgentModel

import results_store

# Matches `s2p/main.py` and `analyze_latency_flops_grid.py`: TF32 matmuls change what is
# being timed, so this has to be the setting evals actually run under.
MATMUL_PRECISION = "high"

# The grid keys that switch on the batched-latency sweep; all or none must be given.
GRID_KEYS = (
	"horizon_values", "iterations_values", "num_samples_values", "num_elites_values",
	"batched_latency_repeats",
)


def _sync(device):
	if torch.device(device).type == "cuda":
		torch.cuda.synchronize(device)


def time_episode(evaluator: RealTaskEvaluator, model: AgentModel, device):
	"""One episode of `RealTaskEvaluator`'s loop without the plan call.

	Returns (reset seconds, [seconds per env step]). The per-step body mirrors the eval's --
	keep the two in step if that loop changes.
	"""
	num_envs = evaluator.num_envs
	action = evaluator.zero_action

	_sync(device)
	start = time.perf_counter()
	obs, done = evaluator.env.reset(), False
	obs = obs.to(device).unsqueeze(0)
	state = model.encoder_model.encode(obs)
	if evaluator.cfg.save_video:
		np.transpose(evaluator.env.render(), (2, 0, 1))
	_sync(device)
	reset_s = time.perf_counter() - start

	return_ = torch.zeros(num_envs, device=device)
	step_s = []
	while not done:
		start = time.perf_counter()
		obs, reward, terminated, truncated, step_info = evaluator.env.step(action.cpu().detach())
		obs = obs.to(device).unsqueeze(0)
		done = episode_boundary(terminated, truncated)
		return_ += reward.to(device).reshape(num_envs)
		if "success" in step_info:
			torch.as_tensor(step_info["success"], device=device).reshape(num_envs).bool()
		if evaluator.cfg.save_video:
			np.transpose(evaluator.env.render(), (2, 0, 1))
		state = model.encoder_model.encode(obs, state, action.unsqueeze(0))
		_sync(device)
		step_s.append(time.perf_counter() - start)
	return reset_s, step_s


def random_batched_obs(task, num_envs, device):
	"""A `(1, num_envs, ...)` random observation, dict-shaped if the task's is."""
	obs_dim = task.observation_dimension
	if isinstance(obs_dim, dict):
		return TensorDict({
			k: torch.randn(size=(1, num_envs, *obs_dim[k]), device=device) for k in obs_dim
		}, batch_size=(1, num_envs), device=device)
	return torch.randn(size=(1, num_envs, *obs_dim), device=device)


def time_batched_plans(model: AgentModel, task, num_envs, repeats, device):
	"""(latencies, peak allocated bytes) of `repeats` plans at `num_envs`, after one warm-up."""
	horizon = model.planner.cfg.horizon
	action_dim = task.action_dimension[-1]
	track_memory = torch.device(device).type == "cuda"
	latencies, peaks = [], []
	for i in range(repeats + 1):
		obs = random_batched_obs(task, num_envs, device)
		prior = torch.randn(size=(horizon, num_envs, action_dim), device=device)
		_sync(device)
		if track_memory:
			torch.cuda.reset_peak_memory_stats(device)
		start = time.perf_counter()
		state = model.encoder_model.encode(obs)
		model.plan(state, prior)
		_sync(device)
		if i == 0:
			continue  # warm-up: kernel selection and allocator growth for this shape
		latencies.append(time.perf_counter() - start)
		if track_memory:
			peaks.append(int(torch.cuda.max_memory_allocated(device)))
	return latencies, peaks


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	torch.set_float32_matmul_precision(MATMUL_PRECISION)
	torch.manual_seed(cfg.seed)
	torch.cuda.manual_seed_all(cfg.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	run_name = cfg.runner.cfg.run_name
	timed_episodes = int(cfg.timed_episodes)
	given = [key in cfg for key in GRID_KEYS]
	if any(given) and not all(given):
		missing = [key for key, g in zip(GRID_KEYS, given) if not g]
		raise ValueError(f"the batched-latency sweep needs all of {GRID_KEYS}; missing {missing}")
	sweep_batched = all(given)

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	evaluator: RealTaskEvaluator = instantiate(cfg.evaluators.real_task_planning)
	task = evaluator.task
	num_envs = evaluator.num_envs
	gpu = results_store.gpu_name(cfg.device)

	# --- Step cost.
	print(f"{run_name} on {gpu}: timing 1 warm-up + {timed_episodes} episodes at {num_envs} env(s), "
	      f"save_video={evaluator.cfg.save_video}")
	time_episode(evaluator, model, cfg.device)
	reset_times, step_times = [], []
	for _ in tqdm(range(timed_episodes), desc="Step cost"):
		reset_s, step_s = time_episode(evaluator, model, cfg.device)
		reset_times.append(reset_s)
		step_times.extend(step_s)

	entry = {
		"gpu": gpu,
		"num_envs": num_envs,
		"save_video": bool(evaluator.cfg.save_video),
		"max_episode_steps": evaluator.cfg.max_episode_steps,
		"episode_steps": len(step_times) // timed_episodes,
		"sim_backend": OmegaConf.select(cfg, "task.cfg.sim_backend"),
		"control_interval_s": float(task.get_control_interval()),
		"reset_times": reset_times,
		"step_times": step_times,
	}
	path = results_store.add_step_cost_entry(run_name, entry)
	print(f"step cost: median {np.median(step_times) * 1e3:.1f} ms/step "
	      f"(p90 {np.percentile(step_times, 90) * 1e3:.1f}), reset {np.median(reset_times) * 1e3:.0f} ms "
	      f"-> {path}")

	# --- Batched plan latency.
	if not sweep_batched:
		return
	if num_envs == 1:
		print("num_envs is 1: batched latency would only repeat `latency`; skipped")
		return

	samples_elites = list(zip(
		[int(n) for n in cfg.num_samples_values], [int(n) for n in cfg.num_elites_values],
		strict=True,
	))
	grid = list(itertools.product(
		[int(h) for h in cfg.horizon_values], [int(i) for i in cfg.iterations_values], samples_elites,
	))
	repeats = int(cfg.batched_latency_repeats)
	planner_cfg = model.planner.cfg

	for horizon, iterations, (num_samples, num_elites) in tqdm(grid, desc=f"Batched latency @ {num_envs} envs"):
		planner_cfg.horizon = horizon
		planner_cfg.iterations = iterations
		planner_cfg.num_samples = num_samples
		planner_cfg.num_elites = num_elites
		if torch.device(cfg.device).type == "cuda":
			torch.cuda.empty_cache()
		try:
			latencies, peaks = time_batched_plans(model, task, num_envs, repeats, cfg.device)
			result = {"gpu": gpu, "num_envs": num_envs, "latencies": latencies, "peak_allocated_bytes": peaks}
			msg = f"{np.median(latencies) * 1e3:.1f} ms, {max(peaks) / 2**30:.2f} GiB" if peaks else f"{np.median(latencies) * 1e3:.1f} ms"
		except torch.cuda.OutOfMemoryError:
			torch.cuda.empty_cache()
			result = {"gpu": gpu, "num_envs": num_envs, "oom": True}
			msg = "out of memory"
		results_store.add_entry(run_name, planner_cfg, "batched_latency", result)
		tqdm.write(f"{results_store.planner_key(planner_cfg)} @ {num_envs} envs: {msg}")

	print(f"batched latency for {len(grid)} configurations -> {results_store.results_path(run_name)}")


if __name__ == "__main__":
	main()
