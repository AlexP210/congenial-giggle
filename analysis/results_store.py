"""Shared results file for the analysis pipeline.

`analyze_latency.py`, `analyze_flops.py` and `analyze_success.py` each test exactly the
planner configuration they are invoked with -- no sweep -- and append one entry to a JSON
file shared by all three scripts and named by `runner.cfg.run_name`:

    agents/squeeze2plan/analysis/results/<run_name>.json

The file is a dict keyed by a canonical string built from the planner's four defining
knobs (`num_samples`, `num_elites`, `horizon`, `iterations`), so repeated invocations of
any of the three scripts against the same configuration -- whether that is two stages of
the same run or the same stage rerun later -- land under the same key rather than
overwriting each other. Each key's value holds the config itself plus one list per stage,
appended to rather than replaced:

    {
      "num_samples=512,num_elites=64,horizon=3,iterations=6": {
        "planner": {"num_samples": 512, "num_elites": 64, "horizon": 3, "iterations": 6},
        "latency": [{"gpu": "NVIDIA L40S", "latencies": [0.0123, ...]}, ...],
        "flops":   [{"flops": 123456789.0}, ...],
        "memory":  [{"gpu": "NVIDIA L40S", "baseline_allocated_bytes": 123456789, "peak_allocated_bytes": [234567890, ...], "peak_reserved_bytes": [268435456, ...]}, ...],
        "success": [{"num_trials": 20, "replan_every": 1, "gpu": "NVIDIA L40S", "successes": [1, 0, 1, ...], "success_at_end": [1, 0, 0, ...], "time_to_success": [42.0, ...], "returns": [12.3, ...]}, ...]
      },
      ...
    }

Writes happen under an flock on a dedicated `.<run_name>.json.lock` file, never the data
file itself -- see `add_entry` -- so two invocations racing to update the same run_name
(another stage, or a rerun of this one) serialize instead of one silently discarding the
other's entry.
"""

import fcntl
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# `batched_latency` is plan latency at an eval's `num_envs` > 1 (`analyze_step_cost.py`),
# kept out of `latency` on purpose: `latencies_for` -- and so `analyze_success.py`'s plan
# budget -- reads every `latency` entry on a GPU, and those must all be single-env.
STAGES = ("latency", "flops", "memory", "success", "batched_latency")

# Per-env-step cost of the real-task eval loop (`analyze_step_cost.py`). Not tied to a planner
# config, so it lives outside `<run_name>.json`, whose every top-level key is one.
STEP_COST_DIR = RESULTS_DIR / "step_cost"

# Real-task success of every checkpoint one training run logged (`analyze_checkpoints.py`).
# Keyed by checkpoint rather than planner config, so it too lives outside `<run_name>.json`.
CHECKPOINT_SWEEP_DIR = RESULTS_DIR / "checkpoint_sweep"
PLANNER_FIELDS = ("num_samples", "num_elites", "horizon", "iterations")


def planner_dict(planner_cfg):
	"""The four config values that key a result, read off a planner's `cfg`."""
	return {field: int(getattr(planner_cfg, field)) for field in PLANNER_FIELDS}


def planner_key(planner_cfg):
	"""Canonical string key for `planner_cfg`, in `PLANNER_FIELDS` order so it is stable
	regardless of the config's own key ordering."""
	values = planner_dict(planner_cfg)
	return ",".join(f"{field}={values[field]}" for field in PLANNER_FIELDS)


def gpu_name(device):
	"""The device name a latency or success measurement is recorded against. Shared by
	`analyze_latency.py` and `analyze_success.py` so the two spell a GPU identically --
	`latencies_for` matches this string exactly, not fuzzily.

	Imports `torch` lazily so the rest of this module -- `load`, `add_entry`,
	`latencies_for` -- stays usable (e.g. from a plotting script, or on the login node)
	without pulling in a GPU-framework dependency nothing else here needs.
	"""
	import torch

	torch_device = torch.device(device)
	if torch_device.type == "cuda":
		return torch.cuda.get_device_name(torch_device)
	return str(device)


def results_path(run_name):
	return RESULTS_DIR / f"{run_name}.json"


def _lock_path(run_name):
	# Deliberately never the data file itself -- see `add_entry`.
	return RESULTS_DIR / f".{run_name}.json.lock"


def add_entry(run_name, planner_cfg, stage, entry):
	"""Append `entry` to `stage`'s list for `planner_cfg` in `<run_name>.json`, creating
	the file or the config's key as needed. Returns the file path.

	Held under an exclusive lock for the whole read-modify-write, so a concurrent call
	against the same file -- another stage, or a rerun of this one -- can't interleave
	with this one and drop an entry.

	The lock is a dedicated `.<run_name>.json.lock` file, never the data file itself, and
	the data file is only ever opened fresh *after* that lock is held. The data write goes
	to a temp file that is then renamed onto the target rather than truncating the target
	in place: a rename is atomic (same directory, POSIX, NFS included), so a kill between
	the two -- Slurm's time limit, `scancel` -- can only ever leave the old complete file
	or the new complete one, never a half-written file that would corrupt every entry
	already on record. But that rename replaces the data file's inode, not its content, so
	a lock taken on the data file's own (pre-open) descriptor would go stale the moment a
	rename lands: the next holder would still be looking at the inode from before, missing
	whatever the previous holder just committed, and its own write would silently discard
	that update rather than build on it. Locking a file that is never replaced is what
	keeps "the lock" and "the current data" from becoming two different objects.
	"""
	if stage not in STAGES:
		raise ValueError(f"stage={stage!r} must be one of {STAGES}")

	def update(results):
		key = planner_key(planner_cfg)
		config_results = results.setdefault(
			key,
			{"planner": planner_dict(planner_cfg), **{s: [] for s in STAGES}},
		)
		config_results.setdefault(stage, []).append(entry)
		return results

	path = results_path(run_name)
	_locked_update(path, _lock_path(run_name), update, empty={})
	return path


def _locked_update(path, lock_path, update, empty):
	"""Read `path` (or `empty` if it does not exist yet), apply `update` to the parsed JSON,
	and write the result back -- under `lock_path`'s exclusive lock, via an atomic rename.
	See `add_entry` for why the lock is a separate file and the write a rename."""
	path.parent.mkdir(parents=True, exist_ok=True)

	with open(lock_path, "a+") as lock_f:
		fcntl.flock(lock_f, fcntl.LOCK_EX)

		# Opened fresh now that the lock is held, never reused across a rename: this is
		# what guarantees the read sees whatever the most recent holder's `os.replace`
		# last committed, rather than the inode that happened to sit at this path when
		# some earlier, unrelated call opened it.
		if path.exists():
			with open(path, "r") as f:
				raw = f.read()
		else:
			raw = ""
		data = update(json.loads(raw) if raw else empty)

		# Same directory as the target, so the rename below can't cross a filesystem
		# boundary (which would make it a copy, not an atomic swap).
		fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
		try:
			with os.fdopen(fd, "w") as tmp:
				json.dump(data, tmp, indent=2)
				tmp.flush()
				os.fsync(tmp.fileno())
			os.replace(tmp_path, path)
		except BaseException:
			os.unlink(tmp_path)
			raise


def step_cost_path(run_name):
	return STEP_COST_DIR / f"{run_name}.json"


def add_step_cost_entry(run_name, entry):
	"""Append one `analyze_step_cost.py` measurement to `step_cost/<run_name>.json`, a JSON
	list, with the same locking and atomic write as `add_entry`. Returns the file path."""
	path = step_cost_path(run_name)
	_locked_update(path, STEP_COST_DIR / f".{run_name}.json.lock",
		lambda entries: [*entries, entry], empty=[])
	return path


def load_step_cost(run_name):
	"""Every step-cost entry recorded for `run_name`, oldest first; `[]` if none."""
	path = step_cost_path(run_name)
	if not path.exists():
		return []
	with open(path, "r") as f:
		raw = f.read()
	return json.loads(raw) if raw else []


def checkpoint_sweep_path(run_name):
	return CHECKPOINT_SWEEP_DIR / f"{run_name}.json"


def add_checkpoint_sweep_entry(run_name, entry):
	"""Append one `analyze_checkpoints.py` evaluation -- one checkpoint of a training run -- to
	`checkpoint_sweep/<run_name>.json`, a JSON list, with the same locking and atomic write as
	`add_entry`. Returns the file path."""
	path = checkpoint_sweep_path(run_name)
	_locked_update(path, CHECKPOINT_SWEEP_DIR / f".{run_name}.json.lock",
		lambda entries: [*entries, entry], empty=[])
	return path


def load_checkpoint_sweep(run_name):
	"""Every checkpoint-sweep entry recorded for `run_name`, oldest first; `[]` if none."""
	path = checkpoint_sweep_path(run_name)
	if not path.exists():
		return []
	with open(path, "r") as f:
		raw = f.read()
	return json.loads(raw) if raw else []


def load(run_name):
	"""The full `{config_key: {"planner": ..., <stage>: [...]}}` dict for `run_name`, or
	`{}` if no results have been recorded for it yet.

	Unlocked: `add_entry` only ever publishes a new version of this file by renaming a
	fully-written temp file onto it, and a rename is atomic, so an open here always lands
	on a complete file -- the one before some writer's update or the one after, never a
	partial one caught mid-write.
	"""
	path = results_path(run_name)
	if not path.exists():
		return {}
	with open(path, "r") as f:
		raw = f.read()
	return json.loads(raw) if raw else {}


def latencies_for(run_name, planner_cfg, gpu):
	"""Every latency this `run_name` has recorded for `planner_cfg` on `gpu` specifically,
	flattened across all matching `analyze_latency.py` entries -- possibly several, from
	separate invocations of the same GPU. Entries recorded on a different GPU are not
	mixed in: latency is hardware-dependent, so a plan budget built from another device's
	measurements would not describe what `analyze_success.py` is actually running on.
	Raises if there are none for this GPU, naming whichever GPUs *are* on record so a
	mismatch (ran latency on one node's GPU, success on another's) is obvious rather than
	silently averaged away."""
	results = load(run_name)
	key = planner_key(planner_cfg)
	entries = results.get(key, {}).get("latency", [])
	latencies = [t for entry in entries for t in entry["latencies"] if entry["gpu"] == gpu]
	if not latencies:
		recorded_gpus = sorted({entry["gpu"] for entry in entries})
		on_record = f"only recorded on {recorded_gpus}" if recorded_gpus else "none recorded"
		raise FileNotFoundError(
			f"no latency entries for {key!r} on {gpu!r} in {results_path(run_name)} "
			f"({on_record}) -- run analyze_latency.py for run_name={run_name} with this "
			f"planner config on {gpu!r} first"
		)
	return latencies
