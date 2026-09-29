"""Fixed-length, frame-stacked, frameskipped windows over per-episode trajectory data.

This is the windowing half of `ManiSkillTrajectoryDataset`, factored out so that a second
recording format can serve the same TensorDict schema through the *same* code rather than a
copy of it. It is deliberately storage-agnostic: a subclass fills `episode_data` with one
nested dict per episode whose leaves are anything supporting `len()` and fancy indexing
along axis 0 — an in-memory numpy array, an `np.memmap` view straight into an h5 file, a
torch tensor on a device, or a lazy decoder onto a video file — then calls
`_index_episodes()`.

What the leaves must agree on is the *alignment* convention, which `_pad_transitions`
establishes: step-indexed leaves (observations, states) have one row per step, while
transition-indexed leaves (actions, rewards, termination flags) have one row fewer and are
NaN-padded at the front, so that after padding index `i` of every leaf refers to the same
moment — "the action/reward/flag that brought the episode to observation `i`", with index 0
carrying the NaN "there was no transition here" marker.

See `ManiSkillTrajectoryDataset` and `PushTTrajectoryDataset` for the two subclasses.
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict


AGGREGATION_BY_KEY = {
    # How each top-level `structure` key turns `frame_skip` primitive steps into the one step of
    # a returned sample. Keyed by name because frameskip is inherently per-quantity.
    #
    # Note what is NOT here: the observations *inside* a macro step are discarded, never gathered.
    # A macro step's observation is the single frame at its boundary -- the one its macro action is
    # chosen from, equivalently the last observation the previous macro step produced. Only
    # actions, rewards and flags aggregate across the skipped steps.
    "obs": "frame_stack",   # that boundary frame, plus the num_frames - 1 previous *boundary* frames
    "action": "concat",     # the frame_skip primitive actions this one macro action stands for
    "reward": "sum",        # return-preserving, matching custom_maniskill_tasks.FrameSkip
    "terminated": "max",    # OR over the chunk for 0/1 flags, and it keeps the NaN pad NaN
    "truncated": "max",
}
DEFAULT_AGGREGATION = "boundary"
"""Any other key (e.g. "state"): the value at the macro step's own boundary index."""


def map_structure(structure, fn):
    """
    Recursively applies fn(path) to every leaf of a (possibly nested) `structure`
    dict, preserving its shape — this is what lets `structure` describe an
    arbitrarily nested TensorDict instead of a flat {nickname: path} mapping.
    Leaves whose path isn't present in this trajectory (fn raises KeyError/TypeError)
    are dropped rather than propagating the error, same as the old
    observations_to_expose behavior.
    """
    if isinstance(structure, dict):
        out = {}
        for key, value in structure.items():
            mapped = map_structure(value, fn)
            if mapped is not None:
                out[key] = mapped
        return out
    try:
        return fn(structure)
    except (KeyError, TypeError):
        return None


def leaf_lengths(node, out):
    if isinstance(node, dict):
        for v in node.values():
            leaf_lengths(v, out)
    else:
        out.append(len(node))
    return out


def pad_transitions(node, full_len):
    """
    Recursively NaN-pad any leaf array one row short of `full_len` at index 0.
    Recordings hold step-indexed fields (obs, env_states, ...) with one row per step and
    transition-indexed fields (actions, rewards, terminated, truncated, ...) with one row
    fewer — ManiSkill writes T of the latter against T+1 of the former; DINO-WM's PushT
    videos hold T frames against T-1 transitions that stay inside the recording.
    Padding by measured length rather than by key name means any path placed anywhere in
    `structure` is handled correctly without the loader needing to know what it's called.
    """
    if isinstance(node, dict):
        return {k: pad_transitions(v, full_len) for k, v in node.items()}
    if len(node) == full_len - 1:
        pad = np.full((1,) + node.shape[1:], np.nan, dtype=np.float32)
        return np.concatenate([pad, node], axis=0)
    return node


def to_tensor(array, device=None):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU, recursing into dicts and
    TensorDicts. float64 is narrowed to float32, and the two unsigned integer types torch
    has no equivalent for are widened to the next signed type that fits.
    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    if isinstance(array, (TensorDict)):
        return TensorDict({k: to_tensor(v, device=device) for k, v in array.items()})
    elif isinstance(array, torch.Tensor):
        ret = array.to(device)
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.from_numpy(array).to(device)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
        ret = torch.from_numpy(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret


class TrajectoryWindowDataset(Dataset):
    """
    Serves per-episode contiguous sub-trajectories as TensorDicts, in the schema the rest of
    this codebase expects (see `OfflineTransitionDataset` in s2p/lib/transition_data.py).

    Frame stacking (`num_frames`) is applied lazily in `__getitem__` rather than eagerly at
    load time: eagerly stacking would multiply the size of the (often image-heavy)
    observation data held in memory by `num_frames`, which is not feasible for large
    datasets.

    Args:
        structure (dict): describes the full shape of the TensorDict returned by
            `__getitem__`. Keys are output field names (arbitrarily nested, e.g. "obs" or
            "state" containing further nicknames); values are whatever the subclass reads as
            a source identifier. Every leaf under "obs" is frame-stacked (see `num_frames`)
            in addition to horizon-windowing; every other leaf is only horizon-windowed. A
            leaf missing from a given trajectory is silently dropped from that trajectory's
            sample rather than raising.
        horizon (int): number of contiguous transitions per returned sample. Windows are
            always drawn from a single episode and never cross episode boundaries.
        num_frames (int): number of past frames to stack into each entry under "obs". The
            beginning of each episode is padded by repeating that episode's first frame, so
            stacking never reaches into a neighboring episode.
        frame_skip (int): how many primitive env steps one step of a returned sample stands
            for, matching `custom_maniskill_tasks.FrameSkip` on the online env, so a policy
            trained on this data acts in the same space it was trained in. Each step's action
            becomes the concatenation of `frame_skip` primitive actions (widening the action
            dimension by that factor), its reward their sum, its flags their OR, and its
            observation the single frame at the step boundary -- the observations in between
            are skipped, not gathered. `num_frames` is orthogonal and still counts *stacked*
            frames, so the stack itself strides by `frame_skip` and never reads a skipped
            frame. 1 (default) leaves every quantity exactly as it is on disk.
        overlap_ratio (float): fraction of overlap between consecutive sub-trajectory windows
            within an episode. 0.0 (default) means windows are back-to-back. Overlap is
            counted in macro steps, so window spacing scales with `frame_skip`.
        batch_device: device onto which a sample's tensors are moved when the dataset is
            indexed.
    """

    def __init__(
        self,
        structure: Dict,
        horizon: int = 1,
        num_frames: int = 1,
        frame_skip: int = 1,
        overlap_ratio: float = 0.0,
        batch_device=None,
    ) -> None:
        assert frame_skip >= 1, f"frame_skip must be >= 1, got {frame_skip}"

        self.structure = structure
        self.horizon = horizon
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.overlap_ratio = overlap_ratio
        self.batch_device = batch_device

        # Filled in by the subclass, then indexed by `_index_episodes`. `episode_metadata` is
        # kept parallel to `episode_data` so per-episode facts that live outside the bulk
        # arrays (a success flag, the source file, reset kwargs) stay addressable by episode
        # index even when loading drops or subsamples episodes.
        self.episode_data: list = []
        self.episode_metadata: list = []
        self.valid_indices: list = []

    def _index_episodes(self) -> None:
        """
        Enumerate the valid (episode_idx, start_step) pairs, in primitive steps.

        A window starting at t has its last macro step at t + (horizon - 1) * frame_skip,
        which has to stay within the episode; `action` (like every transition-indexed leaf)
        is one row longer than the episode's step count after `pad_transitions`.
        """
        stride = max(1, self.horizon - int(self.horizon * self.overlap_ratio)) * self.frame_skip
        self.valid_indices = [
            (ep_idx, t)
            for ep_idx, ep in enumerate(self.episode_data)
            for t in range(
                0, (len(ep["action"]) - 1) - (self.horizon - 1) * self.frame_skip + 1, stride
            )
        ]

    def __len__(self):
        return len(self.valid_indices)

    def _take(self, arr, idx: np.ndarray):
        """Gather `idx` along the time axis, for either an array or a torch tensor.

        Works the same whether `arr` is a real in-memory array, an np.memmap view, or a lazy
        decoder onto a video file: fancy indexing always allocates a fresh, independent array
        — for a memmap view or a video this is the point where the frames actually get read
        from disk, and only those frames.
        """
        idx = np.clip(idx, 0, len(arr) - 1)
        if isinstance(arr, torch.Tensor):
            return arr[torch.from_numpy(idx)]
        return arr[idx]

    def _boundary_steps(self, t: int, H: int) -> np.ndarray:
        """The primitive index each of the H macro steps sits at: t, t+k, t+2k, ..."""
        return t + np.arange(H) * self.frame_skip

    def _stack_obs_window(self, arr, t: int, H: int):
        """
        Build the frame-stacked window for the H macro steps starting at primitive step t.
        Macro step i's value is a stack, along a new axis right after the time axis, of the
        frames at [i - num_frames + 1, ..., i] *macro* steps, i.e. strided by frame_skip, so a
        frame skipped over is never read. Indices are clamped to episode-start (index 0) so
        stacking repeats the episode's first frame instead of crossing into another episode.
        The result has shape (H, num_frames, *arr.shape[1:]) — num_frames is its own
        batch dimension, not folded into the feature/channel dimension.
        """
        n, k = self.num_frames, self.frame_skip
        frame_idx = (
            self._boundary_steps(t, H)[:, None] + (np.arange(-(n - 1), 1) * k)[None, :]
        )  # [H, n]
        frames = self._take(arr, frame_idx.reshape(-1))
        return frames.reshape(H, n, *frames.shape[1:])

    def _aggregate_window(self, arr, t: int, H: int, how: str):
        """Reduce each macro step's `frame_skip` primitive entries to one, per `how`.

        `pad_transitions` has already shifted every transition-indexed leaf by one row, so
        index i means "the action/reward/flag that brought the episode *to* step i", with a NaN
        row at i=0. A macro step's chunk therefore *ends* at its boundary rather than starting
        there: the `frame_skip` entries that brought the episode to that boundary. Reaching back
        before the window start is the same thing the unskipped path does when it takes the action
        that led into the window's first observation, and clamping at index 0 keeps the NaN pad,
        which `sum`/`max` then propagate as the "no transition here" marker for step 0.
        """
        if self.frame_skip == 1:
            # identical result to the paths below, but keeps the common case a plain slice
            return arr[t:t + H]
        if how == "boundary":
            return self._take(arr, self._boundary_steps(t, H))

        chunk_offsets = np.arange(-(self.frame_skip - 1), 1)
        group_idx = self._boundary_steps(t, H)[:, None] + chunk_offsets[None, :]
        group = self._take(arr, group_idx.reshape(-1))
        group = group.reshape(H, self.frame_skip, *group.shape[1:])
        if how == "concat":
            return group.reshape(H, -1)
        if how == "sum":
            return group.sum(1)
        if how == "max":
            # np.max propagates NaN and torch has no dim-wise `max` that returns bare values
            return group.max(1) if isinstance(group, np.ndarray) else group.amax(1)
        raise ValueError(f"unknown aggregation {how!r}")

    def _window_leaf_raw(self, arr, t: int, H: int, how: str):
        """One leaf's window, still as whatever `episode_data` holds: shape (H, ...).

        Split out of `_window_leaf` so the batched path (`__getitems__`) can gather every
        sample's window first and convert once, instead of building a tensor per sample.
        """
        if how == "frame_stack":
            return self._stack_obs_window(arr, t, H)
        return self._aggregate_window(arr, t, H, how)

    # Scalar-per-step leaves (e.g. reward, terminated) need an explicit trailing feature
    # dim of size 1; leaves that already carry one (e.g. a pose vector) don't. This is
    # decided by the array's shape, not by which key it's under, so it applies uniformly
    # to any leaf placed anywhere in `structure`. `_batch_window_structure` applies the
    # same rule one dimension further out, where the leading axis is the batch.
    def _window_leaf(self, arr, t: int, H: int, how: str):
        windowed = to_tensor(self._window_leaf_raw(arr, t, H, how), device=self.batch_device)
        return windowed.unsqueeze(-1) if windowed.ndim == 1 else windowed

    def _window_structure(self, node, t: int, H: int, how: str):
        if isinstance(node, dict):
            windowed = {k: self._window_structure(v, t, H, how) for k, v in node.items()}
            return TensorDict(windowed, batch_size=[H], device=self.batch_device)
        return self._window_leaf(node, t, H, how)

    def stacked_observation(self, episode_index: int, step: int = -1):
        """
        The frame-stacked observation of a single step, addressed by episode rather than
        by window: `__getitem__` can only reach steps that a full horizon-length window
        fits behind, which excludes the end of every episode — exactly where a goal state
        lives.

        Returns the same TensorDict schema as `sample["obs"]`, minus the horizon
        dimension: each leaf is (num_frames, *feature_dims). `step` follows Python
        indexing, so -1 is the terminal observation.
        """
        episode = self.episode_data[episode_index]
        num_steps = max(leaf_lengths(episode["obs"], []))
        step = range(num_steps)[step]  # normalize negatives; raises on out-of-range
        return self._window_structure(episode["obs"], step, 1, AGGREGATION_BY_KEY["obs"])[0]

    def __getitem__(self, idx):
        ep_idx, t = self.valid_indices[idx]
        ep = self.episode_data[ep_idx]
        H = self.horizon

        data = {
            key: self._window_structure(
                value, t, H, AGGREGATION_BY_KEY.get(key, DEFAULT_AGGREGATION)
            )
            for key, value in ep.items()
        }
        return TensorDict(data, batch_size=[H], device=self.batch_device)

    @staticmethod
    def _require_same_keys(nodes, where: str):
        """Every sample in a batch has to carry the same leaves, or they cannot be stacked.

        Single-sample `__getitem__` tolerates a trajectory that is missing a leaf -- it
        simply doesn't appear in that sample. A batch has no such freedom, so say which
        keys disagree rather than failing somewhere inside the stack.
        """
        keys = set(nodes[0].keys())
        for node in nodes[1:]:
            other = set(node.keys())
            if other != keys:
                raise KeyError(
                    f"Episodes in one batch carry different leaves under {where}: "
                    f"{sorted(keys ^ other)} appear in some but not all. Batched loading "
                    "needs every sampled trajectory to have the same structure."
                )
        return list(nodes[0].keys())

    def _batch_window_structure(self, nodes, ts, H: int, how: str):
        """`_window_structure` for a whole batch: gather every sample, then stack once.

        `nodes` is one sample's node per entry, `ts` its window start. Windows come from
        different episodes, so the per-sample gather cannot be vectorized away -- what this
        avoids is the per-sample *wrapping*: one `to_tensor` and one `np.stack` per leaf for
        the entire batch, and one TensorDict per structure level rather than per sample.
        """
        if isinstance(nodes[0], dict):
            keys = self._require_same_keys(nodes, "a batched sample")
            return TensorDict(
                {
                    key: self._batch_window_structure([node[key] for node in nodes], ts, H, how)
                    for key in keys
                },
                batch_size=[len(nodes), H],
                device=self.batch_device,
            )

        windows = [
            self._window_leaf_raw(node, t, H, how) for node, t in zip(nodes, ts)
        ]
        stacked = (
            torch.stack(windows)
            if isinstance(windows[0], torch.Tensor)
            else np.stack(windows)
        )
        stacked = to_tensor(stacked, device=self.batch_device)
        # The batch axis shifts the scalar-per-step test out by one; see `_window_leaf`.
        return stacked.unsqueeze(-1) if stacked.ndim == 2 else stacked

    def __getitems__(self, indices):
        """Build a whole batch at once, returning it already stacked as [B, H, ...].

        torch's map-style fetcher calls this in preference to `__getitem__` when it exists
        (see torch.utils.data._utils.fetch._MapDatasetFetcher), and `Subset` forwards it,
        so a DataLoader picks it up simply by having a `batch_size` -- as long as its
        `collate_fn` passes an already-collated batch straight through.

        The point is allocator pressure, not the reads: the per-sample path built a tensor
        and several TensorDicts for each of `batch_size` samples and then stacked all of
        them, which for a batch of a few hundred image windows is thousands of small
        allocations per step. Here each leaf is gathered into one contiguous array.
        """
        H = self.horizon
        pairs = [self.valid_indices[i] for i in indices]
        eps = [self.episode_data[ep_idx] for ep_idx, _ in pairs]
        ts = [t for _, t in pairs]

        keys = self._require_same_keys(eps, "the batch root")
        data = {
            key: self._batch_window_structure(
                [ep[key] for ep in eps], ts, H, AGGREGATION_BY_KEY.get(key, DEFAULT_AGGREGATION)
            )
            for key in keys
        }
        return TensorDict(data, batch_size=[len(indices), H], device=self.batch_device)
