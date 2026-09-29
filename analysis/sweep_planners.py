"""Sweep planner settings against the task evaluators, flushing every result to disk.

Same shape as `analysis.py`: the config to sweep is chosen on the command line, e.g.

	python analysis/sweep_planners.py --config-name=evaluate_dino_wm_push_cube \
		runner.cfg.run_name=dino_wm_push_cube_sweep \
		evaluators.real_task_planning.cfg.num_episodes=10 \
		evaluators.real_task_planning.cfg.num_envs=10 \
		evaluators.simulation_task_planning.cfg.num_episodes=10 \
		evaluators.simulation_task_planning.cfg.num_envs=10

Everything the evaluators are not told here -- `num_episodes`, `num_envs`, `interim_behaviour`
-- is taken from the composed config, so a sweep is scored under exactly the settings the same
config would be evaluated under by `main.py`. Two exceptions, both module constants below:
`replan_every` is a swept axis, and `MAX_EPISODE_STEPS` fixes the episode length. Where the
config leaves the episode count at the evaluator default, that is 50 episodes at one env, which
over the grid is 10800 episodes -- override it on the command line unless that is really what
is wanted (`train_visual_online_deterministic_*`, for instance, already sets `num_envs: 50` on
both evaluators, so its 50 episodes are a single batch).

The grid is the module constants below -- `HORIZONS` x `REPLAN_EVERY` (capped at the horizon)
x `ITERATIONS` x `POPULATIONS`, 54 points -- and at each point the simulation task planner is
scored under default lighting, then the real (i.e. real-time, planning-costs-env-steps) task
planner under each of `LIGHTING_CONDITIONS`.

Lighting is a property of the env, not of the evaluator, so each condition gets its own task
and therefore its own env (`custom_maniskill_tasks.LIGHTING_PRESETS`). They are built once, on
first use, and reused across the grid -- but that is still one env per (evaluator, lighting)
pair plus the model's own, so this script is heavier on GPU memory than a single evaluation and
`num_envs` should be chosen with that in mind. It is also ManiSkill-only: `lighting` is a
ManiSkill task config key, and only the `-v1.1` task ids accept anything but "default".

Output goes to `analysis/s2p_sweep/<run_name>/`:

	results.jsonl   one JSON object per evaluation, appended and fsync'd as soon as it is
	                finished, so a crash costs at most the evaluation that was running
	results.csv     the same records without the per-episode distributions, rewritten after
	                every evaluation, for loading straight into pandas/a spreadsheet
	config.yaml     the composed config this sweep ran, for provenance

Re-running the same `run_name` picks up where the last invocation stopped: every evaluation
already in results.jsonl is skipped rather than repeated. Delete the folder to start over.

`load_records` / `load_dataframe` at the bottom are what to import tomorrow morning for
plotting.
"""

import csv
import json
import os
import time
import typing
from datetime import datetime

import numpy as np
import torch

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

from s2p.models.agent_model import AgentModel
from s2p.evaluators.simulation_task_evaluator import SimulationTaskEvaluator
from s2p.evaluators.real_task_evaluator import RealTaskEvaluator
from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.lib.seeding import seed_all
from s2p import PROJECT_ROOT


# The planner grid. `POPULATIONS` moves `num_samples` and `num_elites` together rather than
# crossing them: index i means (num_samples=..., num_elites=...), the same pairing
# run_analysis.sh sweeps.
HORIZONS = (1, 3, 5)
ITERATIONS = (1, 3, 5)
POPULATIONS = ((64, 8), (128, 16), (512, 64))

# Env steps executed open-loop between replans, set on both evaluators (it is an evaluator
# setting, not a planner one). Capped at the horizon: a plan is only `horizon` actions long, so
# executing more of it than that is not a thing the evaluators can do. The grid is therefore
# ragged -- horizon 1 admits only `replan_every=1`, horizon 5 admits all three.
REPLAN_EVERY = (1, 3, 5)

# The lighting conditions the real task planner is scored under. "default" is the stock
# lighting every dataset here was recorded with, i.e. what a checkpoint is calibrated for; the
# other two are test-time visual shifts away from it.
LIGHTING_CONDITIONS = ("default", "very-bright", "side")

# What is run at each grid point, in order: (evaluator config key, lighting). The simulation
# evaluator is only run under default lighting -- it measures the planner without the
# real-time penalty, and the lighting axis belongs to the real-time comparison.
EVALUATIONS = (
	("simulation_task_planning", "default"),
	("real_task_planning", "default"),
	("real_task_planning", "very-bright"),
	("real_task_planning", "side"),
)

# The episode length both evaluators are scored under, replacing whatever their configs set
# (they default to 50, and `run_analysis.sh`-style overrides go to the *task*, which these two
# do not read -- see `ManiSkillTask.make_env`). In the task's own units, primitive env steps,
# so at ManiSkill's 20 Hz control rate this is 600 x 0.05 s = 30 seconds of task time per
# episode. It is passed at env construction, so it is fixed for a whole sweep rather than
# swept.
MAX_EPISODE_STEPS = 200

# The grid columns of results.csv, in order; metric columns follow, sorted.
GRID_FIELDS = (
	"run_name",
	"evaluator",
	"lighting",
	"horizon",
	"iterations",
	"num_samples",
	"num_elites",
	"num_episodes",
	"num_envs",
	"replan_every",
	"max_episode_steps",
	"wall_clock_seconds",
	"finished_at",
)

# What identifies an evaluation, and therefore what a resumed run skips.
KEY_FIELDS = (
	"evaluator",
	"lighting",
	"horizon",
	"iterations",
	"num_samples",
	"num_elites",
	"replan_every",
)


def grid_points() -> typing.List[typing.Tuple[int, int, int, int, int]]:
	"""The sweep, as (horizon, replan_every, iterations, num_samples, num_elites) tuples.

	Ragged in `replan_every`, which is capped at the horizon -- so this is what the loop
	iterates and what the progress bar is sized off, rather than a product of the axes.
	"""
	points = []
	for horizon in HORIZONS:
		for replan_every in REPLAN_EVERY:
			if replan_every > horizon:
				continue
			for iterations in ITERATIONS:
				for num_samples, num_elites in POPULATIONS:
					points.append((horizon, replan_every, iterations, num_samples, num_elites))
	return points


def record_key(record: dict) -> tuple:
	"""The identity of one evaluation: its grid point plus what was run there."""
	return tuple(record[field] for field in KEY_FIELDS)


def build_evaluator(cfg: DictConfig, evaluator_name: str, lighting: str) -> OnlineEvaluatorBase:
	"""Instantiate `cfg.evaluators[evaluator_name]` against an env with this lighting.

	The config node is resolved into a detached copy first: `task: ${task}` and the `${device}`
	/ `${seed}` interpolations inside it point at the config root, so they have to be resolved
	while the node is still attached, and the task's `lighting` can only be overridden on a copy
	-- writing it into `cfg.task` would change it for the model's own task too.

	`maniskill_task.get_or_create` memoizes on the task config, so the "default" copy hands back
	the very task the model was built against and only the shifted lightings pay for a new env.
	"""
	node = OmegaConf.create(OmegaConf.to_container(cfg.evaluators[evaluator_name], resolve=True))
	node.task.cfg.lighting = lighting
	# Videos are a wandb thing: nothing here writes them, and over a grid this size they are
	# just arrays of frames held in memory and thrown away.
	node.cfg.save_video = False
	# Read by `make_env` at construction, which is why it is set here and not per evaluation.
	node.cfg.max_episode_steps = MAX_EPISODE_STEPS
	return instantiate(node)


def pin_episode_sequence(evaluator: OnlineEvaluatorBase, seed: int) -> None:
	"""Reset the env's main RNG so every grid point is scored on the same episodes.

	ManiSkill draws each episode's initial state from `_episode_rng`, which is seeded from
	`_main_rng` -- and `_main_rng` is seeded from OS entropy on the first reset of an env and
	then left alone, so an evaluator called once per grid point sees a different draw each time.
	Passing a seed to `reset` re-pins it (`BaseEnv._set_main_rng`), so the episodes the
	evaluator then resets into are the same sequence at every grid point and a difference in
	success rate is the planner's rather than the draw's.
	"""
	evaluator.env.reset(seed=int(seed))


def split_metrics(info: typing.Dict[str, typing.Any]) -> typing.Tuple[dict, dict]:
	"""Split an evaluator's info into JSON-able scalars and per-episode distributions."""
	scalars = {}
	distributions = {}
	for key, value in info.items():
		array = np.asarray(value)
		if array.ndim == 0:
			scalars[key] = array.item()
		elif array.ndim == 1:
			distributions[key] = array.tolist()
		else:
			# Nothing reaches here with `save_video` off; anything that did would be frames.
			continue
	return scalars, distributions


def append_record(results_path: str, record: dict) -> None:
	"""Append one record to the JSONL file and get it onto the disk before returning.

	fsync rather than just flush: the point of writing after every evaluation is that killing
	the run -- or the machine going down overnight -- costs only the evaluation in flight.
	"""
	with open(results_path, "a") as results_file:
		results_file.write(json.dumps(record) + "\n")
		results_file.flush()
		os.fsync(results_file.fileno())


def write_csv(csv_path: str, records: typing.List[dict]) -> None:
	"""Rewrite the flat scalar table. Cheap -- the grid is a hundred-odd rows."""
	metric_fields = sorted({key for record in records for key in record["metrics"]})
	fields = list(GRID_FIELDS) + metric_fields
	with open(csv_path, "w", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		for record in records:
			row = {field: record[field] for field in GRID_FIELDS}
			row.update(record["metrics"])
			writer.writerow(row)


def load_records(results_path: str) -> typing.List[dict]:
	"""Read results.jsonl. A truncated final line -- the run was killed mid-write -- is dropped.

	This is also how a resumed run reads back what it already has, so the drop is what lets it
	rewrite that evaluation rather than carry a half-record forward.
	"""
	if not os.path.exists(results_path):
		return []
	records = []
	with open(results_path) as results_file:
		lines = results_file.readlines()
	for line_number, line in enumerate(lines, start=1):
		line = line.strip()
		if not line:
			continue
		try:
			records.append(json.loads(line))
		except json.JSONDecodeError:
			if line_number != len(lines):
				raise
			print(f"Dropping truncated final line of {results_path} -- it will be re-run.")
	return records


def load_dataframe(results_path: str):
	"""results.jsonl as a flat pandas DataFrame: one row per evaluation, metrics as columns.

	The per-episode distributions stay as list-valued columns, prefixed `distribution/`, so
	error bars can be recomputed from the episodes rather than only read off the recorded SEM.
	"""
	import pandas as pd

	rows = []
	for record in load_records(results_path):
		row = {field: record[field] for field in GRID_FIELDS}
		row.update(record["metrics"])
		row.update({f"distribution/{k}": v for k, v in record["distributions"].items()})
		rows.append(row)
	return pd.DataFrame(rows)


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):

	torch.set_float32_matmul_precision("high")

	# Read before anything is built, so a run launched without it fails in a second rather than
	# after a minute of loading a model. Same for the `???` roots every config here declares:
	# `checkpoint_dir` is only reached deep inside the encoder, so touching them here turns
	# "Missing mandatory value: agent.encoder_model.student.teacher.encoders[0].cfg.checkpoint"
	# into a failure that names the override actually missing from the command line. The job
	# scripts under jobs/ append all four from the sourced machines/*.env.
	run_name = cfg.runner.cfg.run_name
	for root in ("project_root", "data_dir", "checkpoint_dir", "output_dir"):
		OmegaConf.select(cfg, root, throw_on_missing=True)

	# `dinov3_encoder_model` and `dinov3_passthrough_encoder_model` read PROJECT_ROOT from the
	# *environment* at import time, to locate dependencies/dinov3 for `torch.hub.load` -- the
	# `project_root` config key does not reach them, so a run launched without sourcing
	# machines/*.env dies inside the encoder with "expected str, bytes or os.PathLike object,
	# not NoneType". Filling it from the config here makes the command-line override carry.
	# This has to happen before `instantiate` imports either module, which is why it is here
	# rather than beside the imports; the environment wins when it is already set, so a sourced
	# machine env is never overridden.
	project_root = OmegaConf.select(cfg, "project_root")
	if project_root is not None and not os.environ.get("PROJECT_ROOT"):
		print(f"PROJECT_ROOT unset in the environment; using project_root={project_root}")
		os.environ["PROJECT_ROOT"] = str(project_root)

	# Before `instantiate`, as in main.py: Hydra builds the model's weights during
	# instantiation, so seeding any later only makes the run look seeded.
	seed_all(cfg.seed)

	# The model, built and put into the state `EvaluationRunner` would put it in. The runner
	# itself is deliberately not instantiated: it would build every evaluator in the config --
	# including a second copy of these two, plus `planner_convergence`, each with its own env --
	# and this script builds the ones it needs itself, one per lighting condition.
	model: AgentModel = instantiate(cfg.agent)
	model.to(cfg.device)
	model.requires_grad_(False)
	model.eval()

	output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2p_sweep", run_name)
	os.makedirs(output_dir, exist_ok=True)
	results_path = os.path.join(output_dir, "results.jsonl")
	csv_path = os.path.join(output_dir, "results.csv")

	with open(os.path.join(output_dir, "config.yaml"), "w") as config_file:
		config_file.write(OmegaConf.to_yaml(cfg, resolve=True))

	# Anything already recorded under this run name is kept and skipped, so an interrupted
	# sweep is continued by re-running the same command.
	records = load_records(results_path)

	# Episode length is not part of the resume key -- it is one setting for a whole folder, not
	# a swept axis -- so a folder recorded at one `MAX_EPISODE_STEPS` and resumed at another
	# would quietly end up holding two incomparable halves. Refuse instead, and say which.
	for record in records:
		recorded = record["max_episode_steps"] if "max_episode_steps" in record else None
		if recorded != MAX_EPISODE_STEPS:
			raise ValueError(
				f"{results_path} holds evaluations scored at max_episode_steps="
				f"{recorded if recorded is not None else 'unrecorded (written before this was kept)'}"
				f", but this sweep is set to MAX_EPISODE_STEPS={MAX_EPISODE_STEPS}. Resuming would"
				f" mix episode lengths in one file: use a new run_name, or put the constant back."
			)

	completed = {record_key(record) for record in records}
	if completed:
		print(f"Resuming {run_name}: {len(completed)} evaluation(s) already recorded in {results_path}")

	# Built on first use and reused across the grid: an env is expensive, and nothing about it
	# changes as the planner settings do.
	evaluators: typing.Dict[typing.Tuple[str, str], OnlineEvaluatorBase] = {}

	points = grid_points()
	progress = tqdm(total=len(points) * len(EVALUATIONS), desc="Planner sweep")
	progress.update(len(completed))

	for horizon, replan_every, iterations, num_samples, num_elites in points:

		# The planner reads all four of these per plan, so setting them on the live config is
		# all a grid point is -- same as analysis.py. `replan_every` is not among them: it is
		# the evaluator's, and is set on each evaluator below.
		model.planner.cfg.horizon = horizon
		model.planner.cfg.iterations = iterations
		model.planner.cfg.num_samples = num_samples
		model.planner.cfg.num_elites = num_elites

		for evaluator_name, lighting in EVALUATIONS:

			key = (
				evaluator_name, lighting, horizon, iterations, num_samples, num_elites, replan_every
			)
			if key in completed:
				continue

			print(
				f"horizon={horizon} replan_every={replan_every} iterations={iterations} "
				f"num_samples={num_samples} num_elites={num_elites} :: "
				f"{evaluator_name} @ {lighting} lighting"
			)

			if (evaluator_name, lighting) not in evaluators:
				evaluators[(evaluator_name, lighting)] = build_evaluator(
					cfg, evaluator_name, lighting
				)
			evaluator = evaluators[(evaluator_name, lighting)]

			# Read per plan by both evaluators (`SimulationTaskEvaluator.__call__`,
			# `RealTaskEvaluator.__call__`), so setting it on the live config is enough --
			# the evaluator does not have to be rebuilt for it.
			evaluator.cfg.replan_every = replan_every

			# Re-seeded per evaluation rather than once per run: what the planner
			# samples then depends on the settings under test and not on how many
			# evaluations happened to run before it, which is also what makes a
			# resumed sweep agree with an uninterrupted one.
			seed_all(cfg.seed)
			pin_episode_sequence(evaluator, cfg.seed)

			started = time.time()
			info = evaluator(model)
			elapsed = time.time() - started

			scalars, distributions = split_metrics(info)
			record = {
				"run_name": run_name,
				"evaluator": evaluator_name,
				"lighting": lighting,
				"horizon": horizon,
				"iterations": iterations,
				"num_samples": num_samples,
				"num_elites": num_elites,
				"num_episodes": int(evaluator.cfg.num_episodes),
				"num_envs": int(evaluator.cfg.num_envs),
				"replan_every": int(evaluator.cfg.replan_every),
				"max_episode_steps": int(evaluator.cfg.max_episode_steps),
				"wall_clock_seconds": elapsed,
				"finished_at": datetime.now().isoformat(timespec="seconds"),
				"metrics": scalars,
				"distributions": distributions,
			}

			# Flushed here, before the next evaluation starts: this is the whole point
			# of recording per evaluator rather than per grid point.
			records.append(record)
			completed.add(key)
			append_record(results_path, record)
			write_csv(csv_path, records)

			progress.update(1)

	progress.close()
	print(f"Wrote {len(records)} evaluation(s) to {results_path} and {csv_path}")


if __name__ == "__main__":
	main()
