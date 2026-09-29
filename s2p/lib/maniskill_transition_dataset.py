import os
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm
import h5py
import numpy as np
from mani_skill.utils.structs.types import Array, Device

from mani_skill.utils.io_utils import load_json

# The windowing/frameskip machinery lives in trajectory_window_dataset so that the PushT
# recording format goes through exactly the same code. AGGREGATION_BY_KEY and
# DEFAULT_AGGREGATION are re-exported: they used to be defined here, and reading them off
# this module is still the natural way to ask "what does frameskip do to key X".
from s2p.lib.trajectory_window_dataset import (
    AGGREGATION_BY_KEY,
    DEFAULT_AGGREGATION,
    TrajectoryWindowDataset,
    leaf_lengths as _leaf_lengths,
    map_structure as _map_structure,
    pad_transitions as _pad_transitions,
    to_tensor as _to_tensor,
)

def _remove_np_uint16(x: Union[np.ndarray, dict]):
    # uint16 dtype is used to conserve disk space and memory; cast to int32 for torch compatibility
    if isinstance(x, dict):
        return {k: _remove_np_uint16(v) for k, v in x.items()}
    if x.dtype == np.uint16:
        return x.astype(np.int32)
    return x

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def _load_episode(args):
    """
    Reads and preprocesses one trajectory group from the worker's h5py.File.
    Runs in a subprocess: stays on CPU/numpy so results are always picklable,
    device placement happens back in the main process after collection.
    """
    eps, structure, success_only = args
    if success_only:
        assert "success" in eps, "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
        if not eps["success"]:
            return None

    trajectory = load_h5_data(_worker_h5_file[f"traj_{eps['episode_id']}"])
    trajectory = _remove_np_uint16(trajectory)

    # HWC -> CHW for image observations. Frame stacking is deferred to __getitem__.
    ep = _map_structure(structure, lambda path: _to_chw(_get_nested(trajectory, path)))
    full_len = max(_leaf_lengths(ep, []))
    ep = _pad_transitions(ep, full_len)
    return ep

def _init_worker(dataset_file):
    global _worker_h5_file
    _worker_h5_file = h5py.File(dataset_file, "r")

def _to_chw(x):
    if isinstance(x, np.ndarray) and x.ndim == 4:
        return np.transpose(x, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
    return x

def _get_nested(d, path):
    for key in path.split("/"):
        d = d[key]
    return d

def to_tensor(array: Array, device: Optional[Device] = None):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. Thin alias kept for callers that
    import it from here; the implementation is shared with the other trajectory datasets.

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None, which leaves an
            already-placed tensor where it is and puts a numpy array on the CPU.
    """
    return _to_tensor(array, device=device)

def _copy_with_progress(src: Path, dst: Path, size: int, chunk: int = 32 << 20) -> None:
    """Byte-for-byte copy of `src` to `dst`, ticking a progress bar as it goes.

    shutil.copyfile is marginally faster (it can hand the whole transfer to the kernel via
    copy_file_range/sendfile), but staging a multi-GB dataset off a networked filesystem runs
    for minutes, and a job that prints nothing for minutes is indistinguishable from a hung
    one. `mininterval` keeps the log readable when stderr is a file rather than a terminal,
    where tqdm cannot rewrite a line in place and every update becomes another line.
    """
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
        total=size,
        desc=f"Copying {src.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        mininterval=5.0,
    ) as bar:
        buf = bytearray(chunk)
        view = memoryview(buf)
        while True:
            n = fsrc.readinto(buf)
            if not n:
                break
            fdst.write(view[:n])
            bar.update(n)


def stage_to_slurm_tmpdir(data_path) -> Path:
    """
    Stage the dataset on node-local disk ($SLURM_TMPDIR) and return the local copy's path.

    home/project/scratch are networked filesystems, so every page the memmap views fault
    in during training crosses the network -- and in mmap mode (`load=False`) that is every
    page of every sample, for the whole run. One sequential copy up front turns the whole
    job's random reads into local ones.

    Staging is an optimization, never a requirement: whenever it isn't possible this falls
    back to reading `data_path` where it already lives, and prints why.

    This mirrors `_stage_to_slurm_tmpdir` in the dino_wm/TC-WM/dino_bsmpc/sparse_imagination
    copies of maniskill_dset.py; keep the behaviour in step if you touch one.
    """
    data_path = Path(data_path)
    tmpdir = os.environ.get("SLURM_TMPDIR")
    if not tmpdir:
        return data_path

    local_dir = Path(tmpdir)
    size = data_path.stat().st_size

    # Apptainer/Singularity pass the host environment straight through but only mount the
    # paths they were told to, so inside a container $SLURM_TMPDIR is routinely set while
    # pointing at nothing. Don't create it: without a bind mount that would land in the
    # container's own writable layer, which is not the node-local disk we're after.
    if not local_dir.is_dir():
        print(
            f"Not staging data: $SLURM_TMPDIR={tmpdir} is not visible from here (in a "
            f"container, add '--bind $SLURM_TMPDIR'). Reading {data_path} directly.",
            flush=True,
        )
        return data_path

    local = local_dir / data_path.name
    if local.exists() and local.stat().st_size == size:
        print(f"Using dataset already staged at {local}", flush=True)
        return local

    free = shutil.disk_usage(local_dir).free
    if free < size:
        print(
            f"Not staging data: {data_path.name} needs {size / 1e9:.1f} GB but $SLURM_TMPDIR "
            f"has {free / 1e9:.1f} GB free. Reading {data_path} directly.",
            flush=True,
        )
        return data_path

    # Copy to a pid-unique name and rename into place, so a second process on the node
    # (another rank, or a rerun) either sees the complete file or does its own copy --
    # never a half-written one.
    partial = local.with_name(f"{local.name}.{os.getpid()}.partial")
    print(
        f"Copying data: {data_path} -> {local} ({size / 1e9:.1f} GB), this may take a while...",
        flush=True,
    )
    try:
        _copy_with_progress(data_path, partial, size)
        os.replace(partial, local)
    except OSError as e:
        # Out of space, a vanished tmpdir, a concurrent rank filling the disk: none of these
        # are worth killing a GPU job over when the original file is still readable.
        partial.unlink(missing_ok=True)
        print(f"Copying data failed ({e}); reading {data_path} directly.", flush=True)
        return data_path
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print("Copying data: done", flush=True)
    return local


def _view_dataset(raw: np.memmap, dataset: h5py.Dataset) -> np.ndarray:
    """
    Zero-copy view of a contiguous (uncompressed, unchunked) h5py.Dataset, backed by `raw`
    — a single np.memmap covering the whole source file. This reads no data itself: disk
    I/O only happens lazily, a page at a time, once the returned array is actually indexed.
    """
    offset = dataset.id.get_offset()
    if offset is None:
        raise ValueError(
            f"{dataset.name!r} in {dataset.file.filename!r} is chunked and/or compressed and "
            "has no fixed file offset, so it cannot be memory-mapped. Regenerate the file with "
            "tools/preprocess_data_mmap.py, which stores every dataset contiguously."
        )
    return np.ndarray(shape=dataset.shape, dtype=dataset.dtype, buffer=raw, offset=offset)


def _load_episode_view(h5_file: h5py.File, raw: np.memmap, args):
    """
    Mmap-mode counterpart of _load_episode: builds every array in `structure` as an
    np.memmap view (via `raw`) instead of reading it into RAM. Runs single-process/
    in-line rather than in a worker pool — constructing a view is cheap pointer
    arithmetic with no decompression to parallelize, and np.memmap views would defeat
    the purpose of this dataset if pickled wholesale across process boundaries to workers.
    """
    eps, structure, success_only = args
    if success_only:
        assert "success" in eps, "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
        if not eps["success"]:
            return None

    traj = h5_file[f"traj_{eps['episode_id']}"]

    ep = _map_structure(structure, lambda path: _to_chw(_view_dataset(raw, _get_nested(traj, path))))
    full_len = max(_leaf_lengths(ep, []))
    ep = _pad_transitions(ep, full_len)
    return ep


class ManiSkillTrajectoryDataset(TrajectoryWindowDataset):
    """
    Loads ManiSkill trajectory .h5 data and serves it as fixed-length, per-episode
    contiguous sub-trajectories, in the same TensorDict schema expected elsewhere in
    this codebase (see OfflineTransitionDataset in s2p/lib/transition_data.py).

    Only the reading of a ManiSkill recording lives here: the horizon windowing, frame
    stacking and frameskip aggregation are `TrajectoryWindowDataset`'s, shared with the
    other recording formats so their semantics cannot drift apart.

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        structure (dict): describes the full shape of the TensorDict returned by
            __getitem__. Keys are output field names (arbitrarily nested, e.g. "obs" or
            "state" containing further nicknames); values are "/"-separated paths into
            each trajectory's h5 group (e.g. "obs/sensor_data/base_camera/rgb",
            "actions", "env_states/actors/cube"). Every leaf under "obs" is frame-stacked
            (see `num_frames`) in addition to horizon-windowing; every other leaf is only
            horizon-windowed. A leaf path missing from a given trajectory is silently
            dropped from that trajectory's sample rather than raising.
        horizon (int): number of contiguous transitions per returned sample. Windows
            are always drawn from a single episode and never cross episode boundaries.
        num_frames (int): number of past frames to stack into each entry under "obs".
            The beginning of each episode is padded by repeating that episode's first
            frame, so stacking never reaches into a neighboring episode.
        frame_skip (int): how many primitive env steps one step of a returned sample stands
            for, matching `custom_maniskill_tasks.FrameSkip` on the online env, so a policy
            trained on this data acts in the same space it was trained in. Each step's action
            becomes the concatenation of `frame_skip` primitive actions (widening the action
            dimension by that factor), its reward their sum, its flags their OR, and its
            observation the single frame at the step boundary -- the observations in between are
            skipped, not gathered. `num_frames` is orthogonal and still counts *stacked* frames,
            so the stack itself strides by `frame_skip` and never reads a skipped frame.
            1 (default) leaves every quantity exactly as it is on disk.
        fraction (float): fraction of the loaded episodes to keep, sampled without
            replacement. 1.0 (default) keeps all of them, in their original order.
        overlap_ratio (float): fraction of overlap between consecutive sub-trajectory
            windows within an episode. 0.0 (default) means windows are back-to-back.
        load_count (int): the number of trajectories from the dataset file to consider
            loading (before `fraction` subsampling). If -1, all trajectories are considered.
        success_only (bool): whether to skip trajectories that are not successful in the end.
        load (bool): True materializes every array in `structure` into RAM up front, in
            worker processes. False (default) keeps them as np.memmap views straight into
            `dataset_file`: __init__ does no bulk disk I/O, and each __getitem__ call only
            reads the horizon/frame-stack window it actually needs. Resident memory then
            stays bounded regardless of dataset size, at the cost of first-touch disk
            latency (the OS page cache serves repeat touches, e.g. across epochs, at close
            to RAM speed). `load=False` requires `dataset_file` to have been produced by
            tools/preprocess_data_mmap.py, since memmap views need every h5py.Dataset to
            be stored contiguously and uncompressed.
        load_device: device episode tensors are moved to when `load=True`. Ignored (must be
            None) when `load=False` — a memory-mapped array has no single fixed device.
        batch_device: device onto which a sample's tensors are moved when the dataset is indexed.
        num_workers (int): number of worker processes used to read and decompress
            trajectories concurrently when `load=True`. -1 (default) uses
            min(cpu_count, 8); 0 or 1 loads in the main process without spawning a pool.
            Ignored when `load=False`, since building memmap views has no decompression
            to parallelize. Each worker ships its decoded trajectory (obs arrays
            included) back to the main process over IPC, so for image-heavy observations
            more workers can eventually make things *slower* once IPC/pickling saturates
            rather than decompression — tune this per-dataset rather than assuming higher
            is faster.
    """

    def __init__(
        self,
        dataset_file: str,
        json_file: str,
        structure: Dict,
        horizon: int = 1,
        num_frames: int = 1,
        frame_skip: int = 1,
        fraction: float = 1.0,
        overlap_ratio: float = 0.0,
        load_count: int = -1,
        success_only: bool = False,
        load: bool = False,
        load_device=None,
        batch_device=None,
        num_workers: int = -1,
    ) -> None:
        assert load_device is None or load, (
            "load_device is only meaningful when load=True — in mmap mode (load=False) "
            "observation arrays live on disk/OS page cache, not on a fixed device."
        )
        super().__init__(
            structure=structure,
            horizon=horizon,
            num_frames=num_frames,
            frame_skip=frame_skip,
            overlap_ratio=overlap_ratio,
            batch_device=batch_device,
        )

        self.dataset_file = dataset_file
        self.fraction = fraction
        self.load = load

        self.data = h5py.File(dataset_file, "r")
        self.json_data = load_json(json_file)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        if load_count == -1:
            load_count = len(self.episodes)
        candidate_ids = np.arange(load_count)
        n_to_keep = int(self.fraction * load_count)
        episode_ids = (
            candidate_ids if self.fraction >= 1.0
            else np.random.choice(candidate_ids, n_to_keep, replace=False)
        )
        work_items = [
            (self.episodes[int(eps_id)], self.structure, success_only)
            for eps_id in episode_ids
        ]

        if self.load:
            # Each trajectory lives in its own HDF5 group, so reading is inherently
            # per-trajectory (no shared array to slice across groups at once) — but the
            # per-trajectory reads/decompression are independent, so we parallelize them
            # across worker processes instead of doing them one at a time in this process.
            workers = min(os.cpu_count() or 1, 8) if num_workers == -1 else num_workers
            workers = max(0, min(workers, len(work_items)))
            if workers <= 1:
                _init_worker(dataset_file)
                results = [_load_episode(item) for item in tqdm(work_items, desc="Loading Data")]
            else:
                with mp.Pool(workers, initializer=_init_worker, initargs=(dataset_file,)) as pool:
                    results = list(tqdm(
                        pool.imap(_load_episode, work_items),
                        total=len(work_items), desc="Loading Data",
                    ))
        else:
            # A single memmap of the whole file; per-dataset views below are cheap pointer
            # arithmetic into it (see _view_dataset), so there's nothing here worth
            # parallelizing across worker processes — and doing so would require pickling
            # memmap views across process boundaries for no benefit.
            self._raw_mmap = np.memmap(dataset_file, mode="r", dtype=np.uint8)
            results = [
                _load_episode_view(self.data, self._raw_mmap, item)
                for item in tqdm(work_items, desc="Mapping Data")
            ]

        # `episode_metadata` is kept parallel to `episode_data` so per-episode facts that
        # live in the json rather than the h5 (`success`, `episode_id`, the reset kwargs)
        # stay addressable by episode index. Positions do not survive on their own:
        # `fraction` subsamples which episodes are considered and `success_only` drops
        # entries from `results`, so neither `self.episodes` nor `episode_ids` indexes
        # `episode_data`.
        for (eps, _, _), ep in zip(work_items, results):
            if ep is None:
                continue
            if load_device is not None:
                ep = to_tensor(ep, device=load_device)
            self.episode_data.append(ep)
            self.episode_metadata.append(eps)

        self._index_episodes()
