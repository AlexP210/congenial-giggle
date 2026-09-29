"""Frame-stack deduplication in `OnlineTransitionDataset`, against hand-built windows.

Run with `python agents/squeeze2plan/tests/test_online_buffer_packing.py` (no pytest in the project
environment).

The buffer stores a frame-stacked window of `horizon` timesteps as `num_stacked_frames +
horizon - 1` frames instead of `horizon * num_stacked_frames`, exploiting the fact that
`FrameStack` repeats `num_stacked_frames - 1` frames between consecutive timesteps. These
tests check that the round-trip is exact (including the post-reset window, where the stack
holds copies of the initial observation), that non-stacked keys are left alone, and that a
window which is not temporally contiguous is rejected rather than silently mangled.
"""
import torch
from tensordict import TensorDict

from s2p.lib.transition_data import OnlineTransitionDataset

S, H = 3, 4          # num_stacked_frames, horizon
FEAT = (5, 7)        # stand-in for (num_patches, token_length)


def frames(n, offset=0):
    """`n` distinct frames, deterministic so windows built from them can be compared."""
    return torch.arange(offset, offset + n, dtype=torch.float32)[:, None, None].expand(n, *FEAT).clone()


def window(offset=0, after_reset=True):
    """The `[H, S, *FEAT]` window an env with `FrameStack` emits, from frame `offset` on.

    `after_reset` reproduces the wrapper's reset behaviour: the buffer is filled with `S`
    copies of the initial observation, so the first timestep's stack is `[o0, o0, o0]`.
    Mid-episode the window instead spans `H + S - 1` genuinely distinct frames.
    """
    if after_reset:
        observed = frames(H, offset)
        stream = torch.cat([observed[:1].expand(S - 1, *FEAT), observed])
    else:
        stream = frames(H + S - 1, offset)
    return torch.stack([stream[t:t + S] for t in range(H)])


def subtrajectory(obs_window, extra=None):
    source = {("obs", "dino_patch_features"): obs_window,
              "action": torch.randn(H, 4),
              "reward": torch.randn(H, 1)}
    if extra is not None:
        source.update(extra)
    return TensorDict(source=source, batch_size=[H])


def make(**kwargs):
    return OnlineTransitionDataset(capacity=8, horizon=H, num_stacked_frames=S, **kwargs)


def test_round_trip_is_exact():
    """Every stored window comes back bit-identical, from `sample` and `__getitem__` alike."""
    ds = make()
    windows = []
    for i in range(3):
        w = window(offset=100 * i, after_reset=(i == 0))
        windows.append(w)
        ds.add_subtrajectory(subtrajectory(w))

    for i, w in enumerate(windows):
        got = ds[i]["obs"]["dino_patch_features"]
        assert got.shape == w.shape, (got.shape, w.shape)
        assert torch.equal(got, w), f"window {i} did not round-trip"

    batch = ds.sample(16)
    assert batch.batch_size == torch.Size([16, H]), batch.batch_size
    stacked = torch.stack(windows)
    for row in batch["obs"]["dino_patch_features"]:
        assert any(torch.equal(row, w) for w in stacked), "sampled a window never stored"
    print("round trip exact (incl. post-reset window)  OK")


def test_storage_is_actually_smaller():
    """The packed key holds S+H-1 frames per slot; nothing else is touched."""
    ds = make()
    ds.add_subtrajectory(subtrajectory(window()))

    packed = ds.data["obs", "dino_patch_features"]
    assert packed.shape == (8, S + H - 1, *FEAT), packed.shape
    assert ds.packed_keys == frozenset({("obs", "dino_patch_features")}), ds.packed_keys
    # action/reward keep the horizon axis, and the storage batch_size drops to [capacity]
    assert ds.data["action"].shape == (8, H, 4), ds.data["action"].shape
    assert ds.data.batch_size == torch.Size([8]), ds.data.batch_size

    dense = H * S
    print(f"storage {S + H - 1} frames/slot vs dense {dense} "
          f"({dense / (S + H - 1):.2f}x)  OK")


def test_unstacked_keys_are_not_packed():
    """A key that is not under `obs`, or has no frame axis, is stored densely.

    `action` is given dimension S here on purpose: the shape alone cannot distinguish it
    from a frame-stacked observation, so only the structural check keeps it dense.
    """
    ds = make()
    extra = {("obs", "qpos"): torch.randn(H, 9),      # under obs, but no frame axis
             "action": torch.randn(H, S)}            # frame-axis-shaped, but not an obs
    ds.add_subtrajectory(subtrajectory(window(), extra))

    assert ("obs", "qpos") not in ds.packed_keys
    assert "action" not in ds.packed_keys
    assert ds.data["obs", "qpos"].shape == (8, H, 9), ds.data["obs", "qpos"].shape
    print("non-stacked keys stay dense  OK")


def test_no_frame_axis_falls_back_to_dense():
    """`num_stacked_frames=None` reproduces the old dense layout exactly."""
    ds = OnlineTransitionDataset(capacity=8, horizon=H, num_stacked_frames=None)
    w = window()
    ds.add_subtrajectory(subtrajectory(w))

    assert ds.packed_keys == frozenset()
    assert ds.data.batch_size == torch.Size([8, H]), ds.data.batch_size
    assert ds.data["obs", "dino_patch_features"].shape == (8, H, S, *FEAT)
    assert torch.equal(ds[0]["obs"]["dino_patch_features"], w)
    print("num_stacked_frames=None is the dense path  OK")


def test_discontiguous_window_is_rejected():
    """A window straddling an episode boundary must raise, not be silently packed."""
    ds = make()
    ds.add_subtrajectory(subtrajectory(window()))

    # Splice two episodes together: timesteps 0-1 from one, 2-3 from another.
    bad = torch.cat([window()[:2],
                     window(offset=900)[2:]])
    try:
        ds.add_subtrajectory(subtrajectory(bad))
    except ValueError as error:
        assert "does not overlap" in str(error), error
        print("discontiguous window rejected  OK")
    else:
        raise AssertionError("a discontiguous window was accepted")


def test_ring_wraparound():
    """Overwriting past capacity keeps the packed entries consistent."""
    ds = OnlineTransitionDataset(capacity=3, horizon=H, num_stacked_frames=S)
    windows = [window(offset=100 * i) for i in range(5)]
    for w in windows:
        ds.add_subtrajectory(subtrajectory(w))

    assert len(ds) == 3, len(ds)
    # ptr wrapped twice, so slots hold windows 3, 4, 2 in that order
    for slot, expected in zip(range(3), [windows[3], windows[4], windows[2]]):
        assert torch.equal(ds[slot]["obs"]["dino_patch_features"], expected), slot
    print("ring wraparound  OK")


def test_horizon_one_and_single_frame():
    """The degenerate ends of the packing: H=1 (no dedup possible) and S=1 (nothing to dedup)."""
    for horizon, stack in [(1, S), (H, 1)]:
        ds = OnlineTransitionDataset(capacity=4, horizon=horizon, num_stacked_frames=stack)
        stream = torch.arange(horizon + stack - 1, dtype=torch.float32)[:, None, None].expand(-1, *FEAT).clone()
        w = torch.stack([stream[t:t + stack] for t in range(horizon)])
        ds.add_subtrajectory(TensorDict(
            source={("obs", "dino_patch_features"): w, "action": torch.randn(horizon, 4)},
            batch_size=[horizon]))
        assert ds.data["obs", "dino_patch_features"].shape == (4, horizon + stack - 1, *FEAT)
        assert torch.equal(ds[0]["obs"]["dino_patch_features"], w), (horizon, stack)
    print("horizon=1 and num_stacked_frames=1  OK")


def test_parallel_windows_match_sequential_appends():
    """`add_subtrajectories` stores a `[H, num_envs]` batch as `num_envs` ordinary windows.

    What a parallel collector produces is `num_envs` envs stepped in lockstep, i.e. windows
    sharing a time axis. Storing them has to be indistinguishable from appending them one at a
    time -- same entries, same order, same packing -- since a training batch samples windows and
    which env one came from means nothing to it.
    """
    num_envs = 3
    windows = [window(offset=100 * i, after_reset=(i == 0)) for i in range(num_envs)]

    batched = make()
    actions = torch.randn(H, num_envs, 4)
    rewards = torch.randn(H, num_envs, 1)
    batched.add_subtrajectories(TensorDict(
        source={("obs", "dino_patch_features"): torch.stack(windows, dim=1),
                "action": actions,
                "reward": rewards},
        batch_size=[H, num_envs],
    ))

    sequential = make()
    for env_index, w in enumerate(windows):
        sequential.add_subtrajectory(subtrajectory(
            w, extra={"action": actions[:, env_index], "reward": rewards[:, env_index]}
        ))

    assert len(batched) == len(sequential) == num_envs, (len(batched), len(sequential))
    for i in range(num_envs):
        got, expected = batched[i], sequential[i]
        assert torch.equal(got["obs"]["dino_patch_features"], windows[i]), f"env {i} obs"
        for key in ("action", "reward"):
            assert torch.equal(got[key], expected[key]), f"env {i} {key}"
    print("parallel windows match sequential appends  OK")


def test_parallel_append_rejects_a_window_straddling_a_reset():
    """The packing guard still fires per env, so one bad env is not hidden by its neighbours."""
    good = window(after_reset=False)
    bad = torch.stack([window(after_reset=False)[t].roll(1, dims=0) for t in range(H)])
    ds = make()
    try:
        ds.add_subtrajectories(TensorDict(
            source={("obs", "dino_patch_features"): torch.stack([good, bad], dim=1),
                    "action": torch.randn(H, 2, 4)},
            batch_size=[H, 2],
        ))
    except ValueError as error:
        assert "does not overlap" in str(error), error
        print("per-env packing guard fires in a batch  OK")
    else:
        raise AssertionError("a discontiguous env window was accepted")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_round_trip_is_exact()
    test_storage_is_actually_smaller()
    test_unstacked_keys_are_not_packed()
    test_no_frame_axis_falls_back_to_dense()
    test_discontiguous_window_is_rejected()
    test_ring_wraparound()
    test_horizon_one_and_single_frame()
    test_parallel_windows_match_sequential_appends()
    test_parallel_append_rejects_a_window_straddling_a_reset()
    print("\nall packing tests passed")
