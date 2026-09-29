"""
How much does each object in the scene move the encoder's latent?

One bar per item named in `task.cfg.dataset_structure.state`. For each item we hold the whole
rest of the scene -- robot, camera, every other object -- at one fixed state, randomize *only*
that item's pose, render, encode, and measure how far the latent moves. An item whose bar is
near zero is one the encoder is (locally) blind to; an item whose bar dominates is what the
latent is mostly about.

Only the item under test is varied, which is what makes the bars attributable: if the scene were
re-randomized wholesale, every bar would measure the same thing. The fixed background is env 0's
state after the first reset, broadcast across every env.

### Where the poses come from

`sampling: reset` (the default) draws them from the task's *own* reset distribution: the env is
reset repeatedly and each item's pose is lifted out of the resulting state, so every pose tested
is one the task could actually present and the encoder was plausibly trained on. An item the task
does not randomize on reset therefore scores ~0 by construction -- that is reported as a warning,
not silently plotted as "the encoder ignores it".

`sampling: uniform` instead perturbs each item's own initial pose: uniform in x/y within
`position_radius` metres and, with `randomize_orientation`, a uniform yaw about the world z axis.
z is kept, so objects stay on the table. Use this for items the task holds fixed, and to probe
outside the task's own initialization support -- at the cost of testing poses that may be
physically implausible (this sets state directly; nothing settles the scene afterwards).

### What the bar is

For latents z_1..z_N with mean z̄,

    total_variance = sum over latent dims of Var(z)  =  mean_i ||z_i - z̄||^2   (Bessel-corrected)
    bar            = sqrt(total_variance)

i.e. the RMS distance of a latent from the mean latent, in the units the latent itself is in.
Summing the per-dimension variances rather than averaging them keeps the number comparable
across encoders of different widths only after the normalization in the right-hand panel, which
divides by the "all items" bar; the left-hand panel is absolute and is comparable only within one
model.

Two reference bars come for free and are worth reading first:

- **none** -- the same state rendered and encoded N times, with nothing randomized at all. This is
  the noise floor, and any item bar within reach of it is indistinguishable from zero. On a
  deterministic encoder it comes out at ~1e-7 (pure renderer/GPU non-determinism); on a *stochastic*
  one it is the encoder's own sampling noise, and it can be large enough to swamp every item --
  which is the single most useful thing this plot can tell you about such a model.
- **all items** -- every item randomized together, i.e. the scale of the full task distribution.

Usage (same entry points as `analysis.py`, and the same mandatory `runner.cfg.run_name`):

    python analysis/latent_pose_sensitivity.py --config-name=evaluate_dino_wm_push_cube \\
        runner.cfg.run_name=dino_wm_push_cube task.cfg.data_path=null task.cfg.json_path=null \\
        +sensitivity.num_samples=128

`task.cfg.data_path=null` matters for the same reason it does in every evaluation config: a
dataset's stored dimensions otherwise override the env's (see `ManiSkillTask.__init__`).
"""

import os

# Before anything can pull dm_control in transitively: see s2p.evaluators.real_task_evaluator.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import math
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from s2p import PROJECT_ROOT
from s2p.models.agent_model import AgentModel
from s2p.tasks.maniskill_task import ManiSkillTask, ManiSkillWrapper

from custom_maniskill_tasks.wrappers import FrameStack


# This script's own knobs. Not part of any task/model config, so they live here with their
# defaults rather than in `configs/`; override any of them on the command line with
# `+sensitivity.<key>=<value>`.
SENSITIVITY_DEFAULTS = OmegaConf.create({
	# Poses drawn per item. Rounded *up* to a whole number of batches of `task.cfg.num_envs`,
	# since one reset of a batched env yields one independently randomized pose per env.
	"num_samples": 100,
	# "reset" (poses from the task's own reset distribution) or "uniform" (poses perturbed
	# from the item's initial one). See the module docstring.
	"sampling": "reset",
	# "uniform" only: half-extent of the x/y box, in metres, and whether to spin the item.
	"position_radius": 0.1,
	"randomize_orientation": True,
	# The two reference bars. Each costs one more item's worth of renders.
	"include_none_baseline": True,
	"include_all_items": True,
	# Where the figure and the .npz go. None -> analysis/latent_sensitivity/.
	"output_dir": None,
})

KINEMATIC_DIM = 13
"""[position (3), quaternion wxyz (4), linear velocity (3), angular velocity (3)] -- the layout of
an actor's state row, and of the first 13 columns of an articulation's (see `BaseEnv.set_state`)."""

POSE_DIM = 7
"""The position and quaternion at the front of that row: the whole of what a still render sees."""

STATE_PREFIX = "env_states/"
"""What a `dataset_structure` path for a privileged scene state starts with; the rest is the
`get_state_dict()` path, e.g. `actors/cube`."""


# ------------------------------------------------------------------------------------ plumbing

def find_wrapper(env, wrapper_type):
	"""The outermost wrapper of `wrapper_type` in `env`'s stack."""
	while not isinstance(env, wrapper_type):
		if not hasattr(env, "env"):
			raise TypeError(f"No {wrapper_type.__name__} in this env's wrapper stack.")
		env = env.env
	return env


def clone_state(state_dict):
	"""A private, writable copy of the settable half of `get_state_dict()`.

	`controller` is dropped: `BaseEnv.set_state_dict` only forwards `actors`/`articulations` to
	`scene.set_sim_state`, and nothing here steps the env, so the controller's own state never
	comes into it.
	"""
	return {
		group: {name: value.clone() for name, value in state_dict[group].items()}
		for group in ("actors", "articulations")
	}


def broadcast_first_env(state, num_envs):
	"""`state` with env 0's row repeated across every env.

	Every env then shares one background scene, so a bar measures the item under test and not
	the batch it happened to be rendered in.
	"""
	return {
		group: {
			name: value[0:1].repeat(num_envs, *([1] * (value.ndim - 1)))
			for name, value in entries.items()
		}
		for group, entries in state.items()
	}


def resolve_state_path(path):
	"""`("actors", "cube")` from `"env_states/actors/cube"`."""
	if not path.startswith(STATE_PREFIX):
		raise ValueError(
			f"`dataset_structure.state` entry {path!r} is not a privileged scene state: it has "
			f"to start with {STATE_PREFIX!r}, since this analysis sets it through "
			"`env.set_state_dict`."
		)
	group, _, name = path.removeprefix(STATE_PREFIX).partition("/")
	if group not in ("actors", "articulations") or not name:
		raise ValueError(
			f"`dataset_structure.state` entry {path!r} does not name an actor or articulation; "
			f"expected {STATE_PREFIX}actors/<name> or {STATE_PREFIX}articulations/<name>."
		)
	return group, name


def apply_yaw(quaternion, yaw):
	"""`quaternion` (wxyz, `(..., 4)`) pre-rotated by `yaw` about the world z axis."""
	half = 0.5 * yaw
	cos, sin = torch.cos(half), torch.sin(half)
	w, x, y, z = quaternion.unbind(-1)
	return torch.stack([
		cos * w - sin * z,
		cos * x - sin * y,
		cos * y + sin * x,
		cos * z + sin * w,
	], dim=-1)


def perturb_pose(entry, draw, position_radius, randomize_orientation):
	"""`entry` (`(num_envs, >=13)`) with its root pose randomized by the unit `draw` `(num_envs, 3)`.

	x/y uniform within `position_radius` of where it was, z untouched so the item stays on the
	table, and (optionally) a uniform yaw composed onto its orientation. Velocities are zeroed:
	this is a still scene to be rendered, not a trajectory to be continued.
	"""
	out = entry.clone()
	draw = draw.to(out.dtype)
	out[:, 0:2] += (2.0 * draw[:, 0:2] - 1.0) * position_radius
	if randomize_orientation:
		out[:, 3:POSE_DIM] = apply_yaw(out[:, 3:POSE_DIM], 2.0 * math.pi * draw[:, 2])
	out[:, POSE_DIM:KINEMATIC_DIM] = 0.0
	return out


def observe(base_env, wrapper, frame_stack):
	"""The observation the model would see of the scene as it currently stands.

	`get_obs` is exactly what `BaseEnv.step` calls, and it re-renders (`_get_obs_sensor_data`
	updates the render and captures every sensor), so it reflects the state just written rather
	than the last step's frame. `_stacked(..., is_reset=True)` fills the whole frame buffer with
	this one frame, which is the only honest stack for a static scene -- and it leaves no history
	from the previous item behind.
	"""
	return frame_stack._stacked(wrapper._transform_obs(base_env.get_obs()), is_reset=True)


@torch.no_grad()
def encode_current_scene(model, base_env, wrapper, frame_stack, device):
	"""`(num_envs, latent_dim)` for the scene as it currently stands.

	The observation already carries the env axis (`ManiSkillWrapper` keeps ManiSkill's, even at
	one env), so only the time axis is added -- the same `(T=1, B, ...)` layout
	`RealTaskEvaluator` encodes.
	"""
	observation = observe(base_env, wrapper, frame_stack).to(device).unsqueeze(0)
	latent = model.encoder_model.encode(observation)
	return latent.reshape(latent.shape[1], -1).float().cpu()


# ------------------------------------------------------------------------------------- sampling

def draw_reset_pool(env, num_batches, seed):
	"""`num_batches` state dicts, each a batch of independent draws from the task's own reset.

	One reset of a batched env randomizes every env independently, so a batch is `num_envs`
	samples and an entry can be lifted out and used verbatim -- its leading axis already lines up
	with the envs it is about to be written into.
	"""
	pool = []
	for index in tqdm(range(num_batches), "Sampling reset states"):
		env.reset(seed=seed + index)
		pool.append(clone_state(env.unwrapped.get_state_dict()))
	return pool


def draw_uniform_pool(num_batches, num_envs, device, seed):
	"""`num_batches` batches of unit draws, `(num_envs, 3)` each: x, y and yaw, all in [0, 1).

	Drawn once and handed to every item, so two bars differ by *which* item was moved rather than
	by how far it happened to get moved, and a rerun reproduces them whatever order the items are
	visited in.
	"""
	generator = torch.Generator(device=device)
	generator.manual_seed(int(seed))
	return [
		torch.rand((num_envs, 3), generator=generator, device=device)
		for _ in range(num_batches)
	]


def item_states(item_paths, background, pool, batch_index, cfg):
	"""The state to render for one batch: `background`, with `item_paths` randomized.

	`item_paths` is a list so that the "all items" reference bar -- every item randomized at
	once -- is the same code path as a single item.
	"""
	state = clone_state(background)
	for group, name in item_paths:
		if cfg.sampling == "reset":
			state[group][name] = pool[batch_index][group][name].clone()
		elif cfg.sampling == "uniform":
			state[group][name] = perturb_pose(
				background[group][name],
				pool[batch_index],
				cfg.position_radius,
				bool(cfg.randomize_orientation),
			)
		else:
			raise ValueError(
				f"sensitivity.sampling={cfg.sampling!r} is not one of 'reset', 'uniform'."
			)
	return state


def latent_spread(latents):
	"""Summary statistics of how far a set of latents `(N, D)` sits from its own mean.

	`total_variance` is the sum of the per-dimension variances, which is also the mean squared
	distance to the mean latent; `rms` is its square root, i.e. a typical latent's distance from
	the mean, in the latent's own units.

	The standard error is the delta method applied to the per-sample squared distances: `rms` is
	`sqrt(mean(d))`, so `se(rms) = se(mean(d)) / (2 * rms)`. That costs one pass rather than a
	bootstrap over what can be a 300k-dimensional latent.
	"""
	mean = latents.mean(dim=0, keepdim=True)
	squared_distance = ((latents - mean) ** 2).sum(dim=1)
	count = latents.shape[0]
	total_variance = float(squared_distance.sum() / max(count - 1, 1))
	rms = math.sqrt(total_variance)
	standard_error = float(squared_distance.std(unbiased=True) / math.sqrt(count))
	return {
		"total_variance": total_variance,
		"rms": rms,
		"rms_standard_error": standard_error / (2.0 * rms) if rms > 0 else 0.0,
		"mean_latent_norm": float(mean.norm()),
		"latent_dimension": int(latents.shape[1]),
	}


# ------------------------------------------------------------------------------------- plotting

def plot(results, reference, run_name, task_name, sampling, output_path):
	"""Two views of the same numbers: absolute latent movement, and share of the full scene's."""
	names = list(results.keys())
	rms = np.array([results[name]["rms"] for name in names])
	error = np.array([results[name]["rms_standard_error"] for name in names])

	figure, (absolute_axis, relative_axis) = plt.subplots(1, 2, figsize=(12, 5))
	positions = np.arange(len(names))

	absolute_axis.bar(positions, rms, yerr=error, capsize=3, color="#4C72B0")
	if "none" in reference:
		absolute_axis.axhline(
			reference["none"]["rms"], color="#C44E52", linestyle="--", linewidth=1,
			label=f"nothing randomized -- noise floor ({reference['none']['rms']:.3g})",
		)
	if "all" in reference:
		absolute_axis.axhline(
			reference["all"]["rms"], color="#55A868", linestyle=":", linewidth=1.5,
			label=f"all items at once ({reference['all']['rms']:.3g})",
		)
	absolute_axis.set_xticks(positions)
	absolute_axis.set_xticklabels(names, rotation=20, ha="right")
	absolute_axis.set_ylabel("RMS latent deviation")
	absolute_axis.set_title("Latent movement under this item's pose randomization")
	# Headroom for the reference lines, which otherwise sit on the top spine, and for the legend
	# that would then cover them.
	ceiling = max([*rms, *(entry["rms"] for entry in reference.values())])
	absolute_axis.set_ylim(0, ceiling * 1.3)
	if reference:
		absolute_axis.legend(fontsize=8, loc="upper left", framealpha=0.9)

	# Share of the full scene's spread. Squared, i.e. compared as variances, so the numbers are
	# additive in the sense a reader expects: independent items contributing independent latent
	# directions would share out ~100% between them. They generally do not add to 100% -- the
	# encoder is not linear and the items are not independent -- so this is a relative ranking,
	# not a decomposition.
	if "all" in reference and reference["all"]["total_variance"] > 0:
		share = 100.0 * np.array([
			results[name]["total_variance"] for name in names
		]) / reference["all"]["total_variance"]
		relative_axis.bar(positions, share, color="#8172B2")
		relative_axis.set_ylabel("% of the variance of randomizing everything")
		relative_axis.set_title("Share of the full scene's latent spread")
		for position, value in zip(positions, share):
			relative_axis.text(position, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
	else:
		relative_axis.set_title("Share unavailable (no 'all items' reference)")
	relative_axis.set_xticks(positions)
	relative_axis.set_xticklabels(names, rotation=20, ha="right")

	figure.suptitle(f"{run_name} — {task_name} — {sampling} sampling")
	figure.tight_layout()
	figure.savefig(output_path, dpi=300)
	plt.close(figure)


# ----------------------------------------------------------------------------------------- main

@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):

	sensitivity = SENSITIVITY_DEFAULTS.copy()
	if "sensitivity" in cfg:
		sensitivity = OmegaConf.merge(sensitivity, cfg.sensitivity)

	# What RunnerBase would have done. The runner itself is deliberately not instantiated: it
	# would build every evaluator, and each of those stands up its own env -- envs this analysis
	# has no use for, and which all have to agree on one process-wide PhysX backend.
	torch.manual_seed(cfg.seed)
	torch.cuda.manual_seed_all(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	task = instantiate(cfg.task)
	if not isinstance(task, ManiSkillTask):
		raise TypeError(
			f"This analysis randomizes ManiSkill scene objects through `env.set_state_dict`, so "
			f"it only applies to a ManiSkillTask; `task` instantiated to a "
			f"{type(task).__name__}."
		)

	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	run_name = cfg.runner.cfg.run_name

	# The items, in the order the task config lists them.
	items = {
		nickname: resolve_state_path(path)
		for nickname, path in task.cfg.dataset_structure.state.items()
	}

	num_envs = int(task.cfg.num_envs)
	num_batches = math.ceil(int(sensitivity.num_samples) / num_envs)
	num_samples = num_batches * num_envs
	print(
		f"{run_name}: {len(items)} item(s) of {task.cfg.task_name}, {num_samples} pose(s) each "
		f"({num_batches} batch(es) of {num_envs} env(s)), {sensitivity.sampling} sampling."
	)

	env = task.make_env(num_envs=num_envs)
	base_env = env.unwrapped
	wrapper = find_wrapper(env, ManiSkillWrapper)
	frame_stack = find_wrapper(env, FrameStack)

	env.reset(seed=cfg.seed)
	# Every env renders the same background, so the only thing that differs between two renders
	# of one item is that item's own pose.
	background = broadcast_first_env(clone_state(base_env.get_state_dict()), num_envs)

	if sensitivity.sampling == "reset":
		pool = draw_reset_pool(env, num_batches, seed=cfg.seed + 1)
	elif sensitivity.sampling == "uniform":
		pool = draw_uniform_pool(
			num_batches, num_envs, base_env.device, seed=cfg.seed + 1
		)
	else:
		raise ValueError(
			f"sensitivity.sampling={sensitivity.sampling!r} is not one of 'reset', 'uniform'."
		)

	# Each entry is a list of (group, name) to randomize together: one item per bar, plus the
	# two references -- nothing at all, and everything at once.
	plans = {nickname: [path] for nickname, path in items.items()}
	references = {}
	if bool(sensitivity.include_none_baseline):
		references["none"] = []
	if bool(sensitivity.include_all_items):
		references["all"] = list(items.values())

	statistics = {}
	for label, paths in tqdm({**plans, **references}.items(), "Items"):
		latents = []
		for batch_index in range(num_batches):
			state = item_states(paths, background, pool, batch_index, sensitivity)
			base_env.set_state_dict(state)
			latents.append(encode_current_scene(model, base_env, wrapper, frame_stack, cfg.device))
		statistics[label] = latent_spread(torch.cat(latents, dim=0))
		print(f"  {label:>24}: RMS latent deviation {statistics[label]['rms']:.6g}")

	results = {label: statistics[label] for label in plans}
	reference = {label: statistics[label] for label in references}

	# A bar the task's reset never actually moves is a property of the sampling, not of the
	# encoder, and would read on the plot as "the encoder ignores this item".
	if sensitivity.sampling == "reset":
		for nickname, (group, name) in items.items():
			poses = torch.stack([batch[group][name] for batch in pool]).reshape(num_samples, -1)
			# Position and orientation only; a still scene renders the same whatever the
			# velocities are. The tolerance is there because a pose that ManiSkill writes back
			# identically every reset still comes out with ~1e-8 of float32 jitter -- 0.1mm and
			# 1e-4 of a unit quaternion are both far below anything a 224px render resolves.
			if float(poses[:, :POSE_DIM].std(dim=0).max()) < 1e-4:
				print(
					f"WARNING: {nickname} ({group}/{name}) is at the same pose in every reset, "
					f"so its bar measures nothing. Use `+sensitivity.sampling=uniform` to perturb "
					f"it directly instead."
				)

	output_dir = sensitivity.output_dir
	if output_dir is None:
		output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_sensitivity")
	os.makedirs(output_dir, exist_ok=True)
	stem = os.path.join(output_dir, f"{run_name}_{sensitivity.sampling}")

	np.savez(
		f"{stem}.npz",
		labels=np.array(list(statistics.keys())),
		rms=np.array([statistics[label]["rms"] for label in statistics]),
		rms_standard_error=np.array([statistics[label]["rms_standard_error"] for label in statistics]),
		total_variance=np.array([statistics[label]["total_variance"] for label in statistics]),
		mean_latent_norm=np.array([statistics[label]["mean_latent_norm"] for label in statistics]),
		num_samples=num_samples,
		task_name=task.cfg.task_name,
		sampling=sensitivity.sampling,
	)
	plot(results, reference, run_name, task.cfg.task_name, sensitivity.sampling, f"{stem}_{task.cfg.task_name}.png")
	print(f"Wrote {stem}.png and {stem}.npz")


if __name__ == "__main__":
	main()
