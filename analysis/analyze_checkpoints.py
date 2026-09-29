"""Real-task success rate of every checkpoint one TD-MPC2 training run logged, plus a plot of
`real_task_planning/success_once_rate` against training progress with the best one marked.

Driven by the same top-level Hydra configs as `analysis.py`, but only a TD-MPC2 one
(`evaluate_tdmpc2_*.yaml`): every component of `cfg.agent` has to be a `TDMPC2WorldModel`, and
this raises otherwise. The checkpoints are not listed anywhere -- they are every `.pt` file in
the folder holding `tdmpc2_cfg.checkpoint`, which is where TD-MPC2's trainer writes both its
periodic `epoch_<grad_step>.pt` files and `final.pt`. Anything else in that folder is an error,
so a stray file cannot slip into the sweep unordered.

The evaluator is *not* the provided config's `evaluators.real_task_planning`. It is taken from
`REFERENCE_CONFIG`, the online S2P training recipe, so every checkpoint is scored exactly the way
S2P's own online runs score themselves during training (50 episodes over 50 envs, the evaluator's
default 50-step time limit, a plan budget measured on this machine). The evaluation configs set
their own `max_episode_steps: 200` on top, which is exactly what this sidesteps. Its `${task}`,
`${device}` and `${seed}` still resolve against the provided config, so the task, the planner and
the device are the provided config's own.

The agent and the evaluator are built once and reused. Before each checkpoint the weights are
reloaded, every global RNG is reseeded and the env is reset with `cfg.seed`, so each checkpoint is
scored on the same initial states and the same planner noise rather than on whatever episodes
the previous checkpoint left the env at.

Each checkpoint's result is appended to `analysis/results/checkpoint_sweep/<run_name>.json`
(see `results_store.py`) as soon as it finishes, and a rerun with the same `runner.cfg.run_name`
skips checkpoints already on record -- so a kill partway through loses at most the one in flight.
A recorded entry whose evaluator, planner or task config differs from this run's is an error
rather than silently mixed in: choose a new run_name to evaluate under a different config.

The plot, `<run_name>.png` and `.pdf` next to the JSON, is drawn from everything on record, so it
is regenerated in full on every run. The best checkpoint is the one with the highest
success_once_rate; on a tie the later checkpoint wins.

Example:

    python analysis/analyze_checkpoints.py \
        --config-name=evaluate_tdmpc2_push_cube \
        runner.cfg.run_name=Checkpoints_TDMPC2_PushCube \
        hydra.run.dir=/path/to/outputs/hydra \
        data_dir=/path/to/datasets \
        checkpoint_dir=/path/to/pretrained_checkpoints \
        project_root=/path/to/project \
        output_dir=/path/to/outputs \
        device=cuda:0
"""

import os
import re
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from s2p import PROJECT_ROOT
from s2p.evaluators.real_task_evaluator import RealTaskEvaluator
from s2p.lib.seeding import seed_all
from s2p.models.agent_model import AgentModel
from s2p.models.tdmpc2_world_model import TDMPC2WorldModel

import results_store

# Matches `s2p/main.py` -- see `analyze_latency_flops_grid.py`. Here it also sets the plan
# budget the evaluator measures, so it has to be the one training ran under.
MATMUL_PRECISION = "high"

# The config whose `evaluators.real_task_planning` every checkpoint is scored with.
REFERENCE_CONFIG = "train_visual_online_deterministic"

# What `REFERENCE_CONFIG` scores with: 50 episodes, all 50 run at once in one batch. Checked
# against it rather than set on it, so a change to the training recipe fails here instead of
# silently changing what every checkpoint is scored on.
NUM_EPISODES = 50
NUM_ENVS = 50

TDMPC2_TARGET = "s2p.models.tdmpc2_world_model.get_or_create"
AGENT_COMPONENTS = ("encoder_model", "dynamics_model", "reward_model", "value_model", "policy_model")

EPOCH_CHECKPOINT = re.compile(r"epoch_(\d+)\.pt")
FINAL_CHECKPOINT = "final.pt"

METRIC = "episode_success_once_rate"

# Reference palette (dataviz skill): series slot 1 for the sweep, slot 2 for the best checkpoint.
SERIES_COLOR = "#2a78d6"
BEST_COLOR = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def _assert_tdmpc2_agent(cfg):
	"""Every component of `cfg.agent` must be built by TD-MPC2's `get_or_create`."""
	not_tdmpc2 = {
		name: cfg.agent[name]._target_
		for name in AGENT_COMPONENTS
		if cfg.agent[name]._target_ != TDMPC2_TARGET
	}
	if not_tdmpc2:
		raise ValueError(
			f"analyze_checkpoints.py only sweeps TD-MPC2 agents, but these components of "
			f"cfg.agent are not {TDMPC2_TARGET}: {not_tdmpc2}"
		)


def _checkpoint_paths(checkpoint):
	"""Every checkpoint in `checkpoint`'s folder, in training order: `epoch_<N>.pt` by N, then
	`final.pt` (written at the end of training, after the last periodic save)."""
	folder = Path(checkpoint).parent
	if not folder.is_dir():
		raise FileNotFoundError(f"tdmpc2_cfg.checkpoint={checkpoint} is not in an existing folder")

	epochs, final, unrecognised = [], None, []
	for path in folder.iterdir():
		match = EPOCH_CHECKPOINT.fullmatch(path.name)
		if match:
			epochs.append((int(match.group(1)), path))
		elif path.name == FINAL_CHECKPOINT:
			final = path
		else:
			unrecognised.append(path.name)
	if unrecognised:
		raise ValueError(
			f"{folder} holds files that are neither epoch_<N>.pt nor {FINAL_CHECKPOINT}, so they "
			f"cannot be placed in training order: {sorted(unrecognised)}"
		)

	checkpoints = [(step, path) for step, path in sorted(epochs)]
	if final is not None:
		checkpoints.append((None, final))
	if not checkpoints:
		raise FileNotFoundError(f"no checkpoints in {folder}")
	return folder, checkpoints


def _step_label(step):
	if step >= 1e6:
		return f"{step / 1e6:g}M"
	if step >= 1e3:
		return f"{step / 1e3:g}k"
	return f"{step:g}"


def _best_index(rates):
	"""Index of the highest rate, the later one on a tie."""
	return max(range(len(rates)), key=lambda i: (rates[i], i))


def _plot(run_name, entries, checkpoint_folder):
	"""success_once_rate against checkpoint, SEM band, best checkpoint marked. `entries` is in
	training order; `final.pt` has no step of its own, so it is drawn a little past the last
	`epoch_<N>.pt` and labelled."""
	steps = [entry["step"] for entry in entries]
	epoch_steps = [s for s in steps if s is not None]
	last_step = epoch_steps[-1] if epoch_steps else 0
	final_gap = max(0.06 * last_step, 1.0)
	final_x = last_step + final_gap
	xs = np.array([final_x if s is None else s for s in steps], dtype=float)
	rates = np.array([entry[METRIC] for entry in entries])
	sems = np.array([entry[f"{METRIC}_sem"] for entry in entries])
	best = _best_index(rates.tolist())

	fig, ax = plt.subplots(figsize=(9, 4.5))
	ax.fill_between(xs, np.clip(rates - sems, 0, 1), np.clip(rates + sems, 0, 1),
		color=SERIES_COLOR, alpha=0.18, linewidth=0, label="± 1 SEM")
	ax.plot(xs, rates, color=SERIES_COLOR, linewidth=2, marker="o", markersize=3,
		label="success_once_rate")
	ax.axvline(xs[best], color=BEST_COLOR, linewidth=1, linestyle="--", zorder=1)
	ax.plot(xs[best], rates[best], marker="*", markersize=16, color=BEST_COLOR,
		markeredgecolor="white", markeredgewidth=1.5, zorder=5, linestyle="none",
		label=f"best: {entries[best]['checkpoint']}")

	# Keep the callout inside the axes: to the left of the marker when it sits in the right half.
	right_half = xs[best] > (xs.min() + xs.max()) / 2
	ax.annotate(
		f"{entries[best]['checkpoint']}\n{rates[best]:.2f} ± {sems[best]:.2f}",
		xy=(xs[best], rates[best]), xytext=(-12 if right_half else 12, -28),
		textcoords="offset points", ha="right" if right_half else "left", va="top",
		color=TEXT_PRIMARY, fontsize=9,
		bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": GRID_COLOR},
	)

	if any(s is None for s in steps):
		# Regular ticks too close to `final` would collide with its label.
		ticks = [t for t in ax.get_xticks() if 0 <= t <= final_x - final_gap]
		ax.set_xticks([*ticks, final_x])
		ax.set_xticklabels([_step_label(t) for t in ticks] + ["final"])

	ax.set_ylim(-0.02, 1.02)
	ax.set_xlabel("Checkpoint (gradient updates)", color=TEXT_SECONDARY)
	ax.set_ylabel("real_task_planning/success_once_rate", color=TEXT_SECONDARY)
	ax.set_title(f"{run_name}\n{checkpoint_folder}", color=TEXT_PRIMARY, fontsize=10)
	ax.grid(True, color=GRID_COLOR, linewidth=0.8)
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(TEXT_SECONDARY)
	ax.tick_params(colors=TEXT_SECONDARY)
	ax.legend(loc="lower right", frameon=False, fontsize=9)
	fig.tight_layout()

	stem = results_store.checkpoint_sweep_path(run_name).with_suffix("")
	for suffix in (".png", ".pdf"):
		fig.savefig(f"{stem}{suffix}", dpi=300)
	plt.close(fig)
	return best, f"{stem}.png"


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):
	torch.set_float32_matmul_precision(MATMUL_PRECISION)
	seed = int(cfg.seed)
	seed_all(seed)

	run_name = cfg.runner.cfg.run_name
	_assert_tdmpc2_agent(cfg)

	# Swap in the training recipe's evaluator, unresolved, so its interpolations bind to this
	# config's task, device and seed rather than the training config's.
	reference_cfg = compose(config_name=REFERENCE_CONFIG)
	reference_eval_cfg = reference_cfg.evaluators.real_task_planning.cfg
	if (reference_eval_cfg.num_episodes, reference_eval_cfg.num_envs) != (NUM_EPISODES, NUM_ENVS):
		raise ValueError(
			f"{REFERENCE_CONFIG}'s real_task_planning evaluator runs num_episodes="
			f"{reference_eval_cfg.num_episodes} over num_envs={reference_eval_cfg.num_envs}, but this "
			f"sweep expects {NUM_EPISODES} over {NUM_ENVS}"
		)
	with open_dict(cfg):
		cfg.evaluators.real_task_planning = OmegaConf.create(
			OmegaConf.to_container(reference_cfg.evaluators.real_task_planning, resolve=False)
		)

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	# `get_or_create` memoizes on the config, so the components are normally one module; loading
	# each distinct one covers the case where they are not.
	world_models = []
	for name in AGENT_COMPONENTS:
		component = getattr(model, name)
		assert isinstance(component, TDMPC2WorldModel), f"{name} is {type(component).__name__}"
		if not any(component is wm for wm in world_models):
			world_models.append(component)

	checkpoint = world_models[0].parsed_tdmpc2_cfg.checkpoint
	for wm in world_models[1:]:
		if wm.parsed_tdmpc2_cfg.checkpoint != checkpoint:
			raise ValueError(
				f"agent components point at different TD-MPC2 checkpoints "
				f"({checkpoint} vs {wm.parsed_tdmpc2_cfg.checkpoint}), so there is no one run to sweep"
			)
	checkpoint_folder, checkpoints = _checkpoint_paths(checkpoint)

	evaluator: RealTaskEvaluator = instantiate(cfg.evaluators.real_task_planning)
	assert evaluator.num_envs == NUM_ENVS, f"evaluator built {evaluator.num_envs} envs, not {NUM_ENVS}"
	# The plan budget has to be timed here, on `cfg.device`, as it is during training -- never a
	# fixed number carried over from a benchmark on another GPU (e.g. `results_store`'s latencies).
	if evaluator.cfg.steps_needed_to_plan is not None:
		raise ValueError(
			f"real_task_planning.cfg.steps_needed_to_plan={evaluator.cfg.steps_needed_to_plan}, but "
			f"this sweep measures the plan budget on {cfg.device}; it must be null"
		)

	config_record = {
		"evaluator": OmegaConf.to_container(cfg.evaluators.real_task_planning.cfg, resolve=True),
		"planner": OmegaConf.to_container(cfg.planner.cfg, resolve=True),
		"task": cfg.task.cfg.task_name,
	}

	recorded = {}
	for entry in results_store.load_checkpoint_sweep(run_name):
		if entry["path"] not in {str(path) for _, path in checkpoints}:
			raise ValueError(
				f"{results_store.checkpoint_sweep_path(run_name)} records {entry['path']}, which "
				f"is not a checkpoint of {checkpoint_folder} -- choose a new runner.cfg.run_name"
			)
		if entry["config"] != config_record:
			raise ValueError(
				f"{results_store.checkpoint_sweep_path(run_name)} records {entry['checkpoint']} "
				f"under a different evaluator/planner/task config than this run's -- choose a new "
				f"runner.cfg.run_name. Recorded: {entry['config']}; this run: {config_record}"
			)
		recorded[entry["path"]] = entry

	todo = [(step, path) for step, path in checkpoints if str(path) not in recorded]
	print(
		f"{len(checkpoints)} checkpoints in {checkpoint_folder}; {len(recorded)} already on record "
		f"for {run_name}, evaluating {len(todo)}"
	)
	gpu = results_store.gpu_name(cfg.device)
	print(f"plan budget timed on {cfg.device} ({gpu}) at every checkpoint")

	for step, path in tqdm(todo, desc="Checkpoint sweep"):
		for wm in world_models:
			wm.load_from_file(str(path))
		model.requires_grad_(False)
		model.eval()

		seed_all(seed)
		evaluator.env.reset(seed=seed)

		info = evaluator(model)
		num_scored = len(info[f"{METRIC}_distribution"])
		if num_scored != NUM_EPISODES:
			raise RuntimeError(f"{path.name} was scored on {num_scored} episodes, not {NUM_EPISODES}")
		entry = {
			"checkpoint": path.name,
			"path": str(path),
			"step": step,
			"gpu": gpu,
			"config": config_record,
			"steps_needed_to_plan": int(info["steps_needed_to_plan"]),
			METRIC: float(info[METRIC]),
			f"{METRIC}_sem": float(info[f"{METRIC}_sem"]),
			"episode_success_at_end_rate": float(info["episode_success_at_end_rate"]),
			"episode_success_at_end_rate_sem": float(info["episode_success_at_end_rate_sem"]),
			"episode_return": float(info["episode_return"]),
			"episode_return_sem": float(info["episode_return_sem"]),
			"episode_time_to_success": float(info["episode_time_to_success"]),
			"successes_once": info[f"{METRIC}_distribution"].astype(int).tolist(),
			"successes_at_end": info["episode_success_at_end_rate_distribution"].astype(int).tolist(),
			"returns": info["episode_return_distribution"].tolist(),
			"times_to_success": info["episode_time_to_success_distribution"].tolist(),
		}
		results_store.add_checkpoint_sweep_entry(run_name, entry)
		recorded[entry["path"]] = entry
		tqdm.write(
			f"{path.name}: success_once {entry[METRIC]:.2f} ± {entry[f'{METRIC}_sem']:.2f}, "
			f"success_at_end {entry['episode_success_at_end_rate']:.2f}, "
			f"steps_needed_to_plan {entry['steps_needed_to_plan']}"
		)

	entries = [recorded[str(path)] for _, path in checkpoints]

	# The evaluator re-times the planner on every call, so the zero-action steps charged per plan
	# can differ between checkpoints from timing noise alone -- worth knowing before comparing them.
	budgets = sorted({entry["steps_needed_to_plan"] for entry in entries})
	if len(budgets) > 1:
		print(f"WARNING: steps_needed_to_plan varied across checkpoints: {budgets}")

	best, plot_path = _plot(run_name, entries, checkpoint_folder)
	print(
		f"best: {entries[best]['checkpoint']} with success_once_rate "
		f"{entries[best][METRIC]:.2f} ± {entries[best][f'{METRIC}_sem']:.2f} "
		f"-> {results_store.checkpoint_sweep_path(run_name)}, {plot_path}"
	)


if __name__ == "__main__":
	main()
