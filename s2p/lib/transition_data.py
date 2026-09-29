import os
import shutil
import tempfile
import weakref

import torch
import numpy as np
from tensordict import TensorDict, MemoryMappedTensor
from torch.utils.data import Dataset
from enum import Enum

import typing

class OfflineTransitionDataset(Dataset):
    def __init__(
            self, paths:typing.List[str],
            horizon:int = None,
            fraction:float = 1.0,
            overlap_ratio:float = 0.0,
            load_device:str="cpu", 
            batch_device:str="cpu", 
        ):
        self.paths = paths
        self.horizon = horizon
        self.load_device = torch.device(load_device)
        self.batch_device = torch.device(batch_device)
        self.fraction = fraction
        self.overlap_ratio = overlap_ratio

        # Determine number of trajectories in each file and the length of trajectories in each file
        self.num_trajectories = []
        self.trajectory_lengths = []
        self.keys = None
        self.shapes = None
        self.dtypes = None
        for p in self.paths:
            # Number of trajectories & lengths in each file
            data = torch.load(p, weights_only=False)
            self.num_trajectories.append(data["obs"].shape[0])
            self.trajectory_lengths.append(data["obs"].shape[1])

            # Keys in each file (must be the same across files for a consistent dataset)
            if self.keys is None:
                self.keys = list(data.keys())
            elif set(self.keys) != set(list(data.keys())):
                raise ValueError("Not all files have identical keys.")
            
            # Shapes for each key in each file (must be the same across files for a consistent dataset)
            shapes = {
                key: data[key].shape[2:] if len(data[key].shape[2:]) > 0 else (1,) # Make sure even 1-D data has explicit dimension
                for key in data.keys()
            }
            if self.shapes is None:
                self.shapes = shapes
            elif shapes != self.shapes:
                raise ValueError("Not all files have identical shapes for each key.")
            
            # DTypes for each key in each file (must be the same across files for a consistent dataset)
            dtypes = {
                key: data[key].dtype
                for key in self.keys
            }
            if self.dtypes is None:
                self.dtypes = dtypes
            elif dtypes != self.dtypes:
                raise ValueError("Not all files have identical dtypes for each key.")
        
        if self.horizon is None:
            self.horizon = min(self.trajectory_lengths)

        self.trajectory_offset = self.horizon - int(self.horizon*self.overlap_ratio)

        # Get the indices for the trajectories we want to keep
        self.trajectories_to_use = [
            np.random.choice(
                self.num_trajectories[i], 
                int(self.fraction*self.num_trajectories[i]), 
                replace=False
            )
            for i in range(len(self.paths))
        ]
        self.start_indices = [
            np.arange(0, self.trajectory_lengths[i]-self.horizon+1, self.trajectory_offset)
            for i in range(len(self.paths))
        ]
        self.n_subtrajectories = [
            int(self.fraction*self.num_trajectories[i]) * len(self.start_indices[i])
            for i in range(len(self.paths))
        ]

        self.length = sum([
            len(self.trajectories_to_use[i]) * len(self.start_indices[i]) for i in range(len(self.paths))
        ])

        # Allocate the main data tensordict
        self.data = TensorDict(
            source={
                key: torch.empty(
                    size=(self.length, self.horizon, *self.shapes[key]), dtype=self.dtypes[key]
                )
                for key in self.keys
            },
            device=self.load_device,
            batch_size=[self.length, self.horizon]
        )
        self.load()

    def load(self):
        t_offsets = torch.arange(self.horizon)
        c = 0
        for file_idx, path in enumerate(self.paths):
            data = torch.load(path, weights_only=False)
            traj_indices = self.trajectories_to_use[file_idx]
            start_indices = torch.tensor(self.start_indices[file_idx])
            n_start = len(start_indices)
            time_idx = start_indices[:, None] + t_offsets[None, :]  # [n_start, H]

            for key in self.keys:
                key_data = data[key]  # [N_traj, T, *D]
                cc = c
                for ti in traj_indices:
                    windows = key_data[ti][time_idx]  # [n_start, H, *D]
                    if windows.ndim == 2:
                        windows = windows.unsqueeze(2)
                    self.data[key][cc:cc+n_start] = windows.to(self.load_device)
                    cc += n_start

            c += len(traj_indices) * n_start
            del data

    def __getitem__(self, idx):
        return self._getitem_loaded(idx)

    def _getitem_loaded(self, idx):
        return TensorDict(
            source={key: self.data[key][idx] for key in self.keys},
            device=self.batch_device,
            batch_size=[self.horizon],
        )

    def save(self, filepath):
        torch.save(self.data, filepath)

    def __len__(self):
        return self.length
    
class OfflineTransitionDatasetDEBUG(OfflineTransitionDataset):
    def __getitem__(self, idx):
        return self._getitem_loaded(0)


class OnlineTransitionDataset(Dataset):
    """
    Online dataset (replay buffer style).
    Stores transitions in a TensorDict and supports appending during runtime.

    Frame-stacked observations are stored deduplicated when `num_stacked_frames` says the
    env stacks frames on a dedicated axis; see `_init_storage` for what that buys and
    `_packed_window` for the identity it rests on.

    Args:
        capacity (int): number of horizon-length windows the ring buffer holds.
        horizon (int): length of each stored window.
        load_device: device the storage tensors live on. Must be "cpu" when
            `storage_dir` is set.
        batch_device: device a sample's tensors are moved to when the buffer is indexed.
        dtype: if given, floating-point entries are stored at this dtype instead of the
            dtype they arrive with.
        num_stacked_frames (int): number of frames the env's `FrameStack` stacks on a
            dedicated observation axis, which is what lets a window be stored
            deduplicated. None (default) stores every window densely, and is the only
            correct setting for observations with no frame axis to deduplicate along.
        storage_dir (str): directory to back the storage with memory-mapped files
            instead of anonymous RAM. None (default) allocates in RAM. With a directory,
            `_init_storage` reserves one sparse file per key under a fresh subdirectory
            of it, so resident memory stays bounded by whatever the OS keeps in page
            cache rather than by `capacity` — at the cost of disk I/O on eviction, and
            of needing the full reservation (printed at init) free on that filesystem.
            Point it at real disk: a tmpfs mount (`/dev/shm`, and `/tmp` on some
            distributions) is RAM, so mapping into one saves nothing.
    """

    def __init__(
        self,
        capacity: int,
        horizon: int = 1,
        load_device="cpu",
        batch_device="cpu",
        dtype=None,
        num_stacked_frames: typing.Optional[int] = None,
        storage_dir: typing.Optional[str] = None,
    ):
        self.capacity = capacity
        self.horizon = horizon
        self.load_device = torch.device(load_device)
        self.batch_device = torch.device(batch_device)
        self.dtype = dtype
        self.num_stacked_frames = num_stacked_frames
        self.storage_dir = storage_dir

        if num_stacked_frames is not None and num_stacked_frames < 1:
            raise ValueError(
                f"num_stacked_frames must be >= 1 or None, got {num_stacked_frames}"
            )

        assert storage_dir is None or self.load_device.type == "cpu", (
            f"storage_dir needs load_device='cpu' — a memory-mapped buffer is backed by a "
            f"file, so it cannot be allocated on {self.load_device}."
        )

        self._size = 0
        self._ptr = 0

        # storage metadata (initialized lazily once we see the first transition)
        self.keys = None
        self.shapes = None
        self.dtypes = None
        self.data = None
        # Subset of `self.keys` stored as `num_stacked_frames + horizon - 1` frames rather
        # than `horizon * num_stacked_frames`; empty when `num_stacked_frames` is None.
        self.packed_keys = frozenset()
        # [horizon, num_stacked_frames] gather that rebuilds the dense window; built once.
        self._unpack_index = None

        # Set by `_open_storage_dir` when the first transition arrives (see below).
        self._storage_path = None
        self._storage_cleanup = None

    def _open_storage_dir(self) -> str:
        """Create, and arrange the removal of, the directory holding this buffer's files."""
        os.makedirs(self.storage_dir, exist_ok=True)
        # A fresh subdirectory per buffer: concurrent runs (and the trainer's buffer
        # alongside an evaluator's) share `storage_dir` without colliding on file names.
        path = tempfile.mkdtemp(prefix="online_buffer_", dir=self.storage_dir)
        # These files are scratch, not a checkpoint — nothing reads them back once the run
        # ends, and they are tens of GB, so tie their lifetime to this object. finalize()
        # rather than __del__ so the removal still runs at interpreter shutdown, and
        # ignore_errors so a half-written directory cannot turn teardown into a crash.
        self._storage_cleanup = weakref.finalize(self, shutil.rmtree, path, True)
        return path

    def _is_packable(self, key, value: torch.Tensor) -> bool:
        """
        Whether `key` is a frame-stacked observation whose window can be deduplicated.

        Three conditions, all necessary:

        - The key lives under `obs`. Frame stacking is applied by the env's `FrameStack`
          wrapper to the observation and nothing else, so `action`/`reward`/`terminated`/
          `truncated` are never stacked. Checking this structurally rather than inferring it
          from shape is what rules out a false positive on, say, an action whose dimension
          happens to equal `num_stacked_frames`.
        - It carries the frame axis, i.e. `[horizon, num_stacked_frames, ...]`. An
          observation leaf that the wrapper did not stack (or a state-based task, where
          `num_stacked_frames` is None) fails this and is stored densely.
        - Consecutive timesteps actually overlap by `num_stacked_frames - 1` frames. This is
          the identity the packing rests on, so it is checked rather than assumed.
        """
        if self.num_stacked_frames is None:
            return False

        key_path = (key,) if isinstance(key, str) else key
        if key_path[0] != "obs":
            return False

        if value.ndim < 2 or value.shape[1] != self.num_stacked_frames:
            return False

        return bool(torch.equal(value[:-1, 1:], value[1:, :-1]))

    def _packed_window(self, value: torch.Tensor) -> torch.Tensor:
        """
        `[horizon, num_stacked_frames, ...]` -> `[num_stacked_frames + horizon - 1, ...]`.

        `FrameStack` repeats `num_stacked_frames - 1` frames between consecutive timesteps,
        so a dense window stores each frame up to `num_stacked_frames` times over. Keeping
        the full stack at t=0 (which carries the lead-in frames, and after a reset is
        `num_stacked_frames` copies of the initial observation) plus only the newest frame
        of every later timestep is lossless, and is what `_unpack` inverts.

        Exact only while every window handed to `add_subtrajectory` is temporally
        contiguous. Both producers -- `OnlineTrainer.generate_subtrajectory` and
        `OnlineWrapper.generate_episode` -- restart their window at episode boundaries, so
        no window ever straddles a reset. `_append_window` re-checks the overlap identity
        rather than trusting that from a distance.
        """
        return torch.cat([value[0], value[1:, -1]], dim=0)

    def _unpack(self, value: torch.Tensor, batched: bool) -> torch.Tensor:
        """Invert `_packed_window`, over a leading batch axis when `batched`."""
        if self._unpack_index is None or self._unpack_index.device != value.device:
            self._unpack_index = (
                torch.arange(self.horizon, device=value.device)[:, None]
                + torch.arange(self.num_stacked_frames, device=value.device)[None, :]
            )
        return value[:, self._unpack_index] if batched else value[self._unpack_index]

    def _slot_shape(self, key):
        """Shape of one buffer entry for `key`, packed or dense."""
        if key in self.packed_keys:
            return (self.num_stacked_frames + self.horizon - 1, *self.shapes[key][1:])
        return (self.horizon, *self.shapes[key])

    def _empty_storage_tensor(self, key) -> torch.Tensor:
        """Allocate one key's `[capacity, *slot_shape]` storage, in RAM or on disk."""
        shape = (self.capacity, *self._slot_shape(key))
        if self.storage_dir is None:
            return torch.empty(size=shape, dtype=self.dtypes[key], device=self.load_device)

        # Nested keys arrive as tuples ("obs", "rgb"); flatten them into one file name.
        name = "-".join(key) if isinstance(key, tuple) else key
        # The file is created at its full size but sparse, so it only consumes blocks for
        # the windows actually written — a buffer that never fills never costs full price.
        return MemoryMappedTensor.empty(
            shape,
            dtype=self.dtypes[key],
            filename=os.path.join(self._storage_path, f"{name}.memmap"),
        )

    def _init_storage(self, td: TensorDict):
        """Initialize TensorDict storage based on the first horizon-sized sample."""
        # include_nested/leaves_only so composite observations (e.g. a nested
        # "obs" TensorDict with per-camera/sensor keys) get one storage buffer
        # per leaf tensor instead of one (mismatched) buffer per top-level key.
        self.keys = list(td.keys(include_nested=True, leaves_only=True))
        self.shapes = {}
        self.dtypes = {}
        packed_keys = set()

        for key in self.keys:
            value = td[key]
            if value.ndim == 1:
                shape = (1,)
            else:
                shape = value.shape[1:] if len(value.shape[1:]) > 0 else (1,)
            if self.dtype is not None and torch.is_floating_point(value):
                dtype = self.dtype
            else:
                dtype = value.dtype

            self.shapes[key] = tuple(shape)
            self.dtypes[key] = dtype
            if self._is_packable(key, value):
                packed_keys.add(key)

        self.packed_keys = frozenset(packed_keys)

        if self.storage_dir is not None:
            self._storage_path = self._open_storage_dir()

        # A packed key drops `horizon * num_stacked_frames` frames to
        # `num_stacked_frames + horizon - 1` -- 2x at horizon 4, 2.25x at horizon 6 with 3
        # stacked frames -- which is what keeps a buffer of cached DINO features affordable.
        # Its slot therefore has no `horizon` axis at dim 1, so the storage `batch_size` can
        # only claim the leading `capacity`. The one direct reader of `.data`, `_gather` in
        # `evaluators/tsne_evaluator.py`, guards on `len(storage.batch_size) >= 2` and falls
        # back to per-item indexing.
        self.data = TensorDict(
            source={key: self._empty_storage_tensor(key) for key in self.keys},
            device=self.load_device,
            batch_size=[self.capacity] if self.packed_keys else [self.capacity, self.horizon],
        )

        if self.storage_dir is not None:
            reserved = sum(
                self.data[key].numel() * self.data[key].element_size() for key in self.keys
            )
            print(
                f"Memory-mapped replay buffer: reserved {reserved / 2**30:.1f} GiB "
                f"({self.capacity} windows) under {self._storage_path}"
            )

    def _append_window(self, td_window: TensorDict):
        if self.data is None:
            self._init_storage(td_window)

        i = self._ptr
        for key in self.keys:
            value = td_window[key]
            value = value.to(device=self.load_device, dtype=self.dtypes[key])
            if key in self.packed_keys:
                # Cheap against a window that cost `horizon` env steps to collect, and the
                # only thing standing between a window that straddles an episode boundary
                # and a buffer entry that silently decodes to frames the env never produced.
                if not torch.equal(value[:-1, 1:], value[1:, :-1]):
                    raise ValueError(
                        f"Window for '{key}' does not overlap by "
                        f"{self.num_stacked_frames - 1} frames between consecutive "
                        "timesteps, so it cannot be stored deduplicated. Either it is not "
                        "temporally contiguous (does it straddle an episode boundary?) or "
                        "num_stacked_frames does not match the env's frame stack."
                    )
                value = self._packed_window(value)
            self.data[key][i].copy_(value)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_subtrajectory(self, subtrajectory_td: TensorDict):
        """
        Add one horizon-length window, whose batch size is `[horizon]`.
        """
        self._append_window(subtrajectory_td)

    def add_subtrajectories(self, subtrajectories_td: TensorDict):
        """
        Add one window per env from a `[horizon, num_envs]` batch, in env order.

        What a parallel collector produces: `num_envs` envs stepped in lockstep for `horizon`
        steps is `num_envs` windows sharing a time axis. They are stored as separate entries
        because that is what they are -- a training batch samples windows, and which env a
        window came from means nothing to it.

        Sliced apart here rather than taught to the storage layer, so that everything below
        keeps reading dim 0 as time: `_init_storage` sizes each key off `value.shape[1:]` and
        `_append_window` copies one `[horizon, ...]` window into one buffer row, and each slice
        handed on is exactly that window.
        """
        num_envs = subtrajectories_td.batch_size[1]
        for env_index in range(num_envs):
            self._append_window(subtrajectories_td[:, env_index])

    def __len__(self):
        return self._size

    def _to_batch(self, stored: TensorDict, batched: bool) -> TensorDict:
        """
        Move one or more stored entries to `batch_device` and rebuild the dense windows.

        In that order deliberately: unpacking on the batch device means the host->device
        transfer carries the packed frames only, so the saving on the copy is the same
        factor as the saving in the buffer.
        """
        stored = stored.to(self.batch_device)
        if not self.packed_keys:
            return stored

        leading = stored.batch_size[:1] if batched else ()
        return TensorDict(
            source={
                key: self._unpack(stored[key], batched) if key in self.packed_keys
                else stored[key]
                for key in self.keys
            },
            device=self.batch_device,
            batch_size=[*leading, self.horizon],
        )

    def __getitem__(self, idx):
        if self.data is None:
            raise IndexError

        if idx >= self._size:
            raise IndexError

        return self._to_batch(self.data[idx], batched=False)

    def sample(self, batch_size: int):
        """Uniform random sampling (more typical than Dataset indexing)."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        idxs = torch.randint(0, self._size, (batch_size,))
        return self._to_batch(self.data[idxs], batched=True)

    def save(self, path):
        """
        Save the currently valid portion of the buffer as a TensorDict.

        Written in stored (i.e. packed, when packing is on) form: unpacking the whole
        buffer to save it would need `num_stacked_frames` times the memory that packing
        exists to avoid. `num_stacked_frames` is saved alongside so the layout is
        self-describing.
        """
        if self.data is None:
            raise ValueError("Cannot save an empty buffer.")

        torch.save(
            {
                "data": self.data[:self._size],
                "horizon": self.horizon,
                "num_stacked_frames": self.num_stacked_frames,
                "packed_keys": sorted(self.packed_keys),
            },
            path,
        )

if __name__ == "__main__":
    task_index = 24
    task_name = "tdmpc2-cheetah-jump"
    action_ndim = 6
    obs_ndim = 17

    dataset = OfflineTransitionDataset(
        paths=[f"/path/to/tdmpc2_data/datasets/mt30/chunk_{i}.pt" for i in range(4)], fraction=1.0
    )
    tensordict = dataset.data
    mask_indices = tensordict["task"][:, 0, 0] == task_index
    filtered = tensordict[ mask_indices ]
    filtered["action"] = filtered["action"][...,:action_ndim]
    filtered["obs"] = filtered["obs"][...,:obs_ndim]
    torch.save(filtered, f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}.pt")
    dataset = OfflineTransitionDataset(
        paths=[f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}.pt"], fraction=1.0
    )
    tensordict = dataset.data
    validation_mask = torch.rand(tensordict.shape[0]) < 0.2
    validation = tensordict[validation_mask]
    train = tensordict[~validation_mask]
    torch.save(validation, f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}-validation.pt")
    torch.save(train, f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}-training.pt")
    training_dataset = OfflineTransitionDataset(
        paths=[f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}-training.pt"], fraction=1.0
    )
    print("Average Reward: ", training_dataset.data[:,1:]["reward"].mean().item(), training_dataset.data[:,1:]["reward"].std().item())
    validation_dataset = OfflineTransitionDataset(
        paths=[f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}-validation.pt"], fraction=1.0
    )
    print(training_dataset.data.shape)
    print(validation_dataset.data.shape)
    # training_dataset = OfflineTransitionDataset(
    #     paths=[f"/path/to/tdmpc2_data/datasets/mt30/mt30-{task_name}-training.pt"], fraction=1.0
    # )
    # dataset = OfflineTransitionDataset(
    #     paths=[f"/path/to/tdmpc2_data/datasets/mt30/chunk_{i}.pt" for i in range(4)], fraction=1.0
    # )
    # print(training_dataset.data["task"].shape)
    # print(dataset.data["task"].shape)

