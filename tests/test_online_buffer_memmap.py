"""Memory-mapped `OnlineTransitionDataset` storage: parity with RAM, residency, cleanup.

Run with `python agents/squeeze2plan/tests/test_online_buffer_memmap.py` (no pytest in the project
environment).

The point of `storage_dir` is that resident memory stops scaling with `capacity`, so the
tests check the two things that could silently undo that — files that are not sparse, and
reads that materialize more than the requested window — alongside plain read-back parity.
"""
import gc
import math
import os
import shutil
import tempfile

import torch
from tensordict import TensorDict

from s2p.lib.transition_data import OnlineTransitionDataset

CAPACITY = 8
HORIZON = 3
# A stand-in for the pixel observation that dominates the real buffer. Big enough that one
# slot spans several filesystem blocks, which is what makes the sparseness check meaningful.
IMG = (3, 64, 64)


def window(seed: int) -> TensorDict:
    """One horizon-length window whose every entry is a function of `seed`."""
    g = torch.Generator().manual_seed(seed)
    return TensorDict(
        source={
            ("obs", "rgb"): torch.randint(0, 255, (HORIZON, *IMG), dtype=torch.uint8, generator=g),
            ("obs", "proprio"): torch.randn(HORIZON, 5, generator=g),
            "action": torch.randn(HORIZON, 2, generator=g),
            # Trailing 1, matching the `unsqueeze(-1)` the collector applies to scalars.
            "reward": torch.randn(HORIZON, 1, generator=g),
        },
        batch_size=[HORIZON],
    )


def make(storage_dir):
    return OnlineTransitionDataset(
        capacity=CAPACITY, horizon=HORIZON, load_device="cpu", batch_device="cpu",
        storage_dir=storage_dir,
    )


def test_matches_ram_buffer():
    """A memmap-backed buffer reads back exactly what an in-RAM one does, wrap included."""
    root = tempfile.mkdtemp(prefix="s2p_test_memmap_")
    try:
        ram, mmap = make(None), make(root)
        # More windows than capacity, so the ring pointer wraps and overwrites slots 0-3.
        for seed in range(CAPACITY + 4):
            ram.add_subtrajectory(window(seed))
            mmap.add_subtrajectory(window(seed))

        assert len(ram) == len(mmap) == CAPACITY, (len(ram), len(mmap))
        for i in range(CAPACITY):
            a, b = ram[i], mmap[i]
            for key in ram.keys:
                assert torch.equal(a[key], b[key]), (i, key)

        # Slot 0 holds window 8 (the wrap), not window 0.
        assert torch.equal(mmap[0]["action"], window(CAPACITY)["action"])

        idxs = torch.arange(CAPACITY)
        assert torch.equal(ram.data[idxs]["obs", "rgb"], mmap.data[idxs]["obs", "rgb"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_storage_is_on_disk_and_sparse():
    """Files exist, are reserved at full size, and cost blocks only for written slots."""
    root = tempfile.mkdtemp(prefix="s2p_test_memmap_")
    try:
        buf = make(root)
        buf.add_subtrajectory(window(0))

        files = {
            name: os.path.join(buf._storage_path, name)
            for name in os.listdir(buf._storage_path)
        }
        assert set(files) == {"obs-rgb.memmap", "obs-proprio.memmap", "action.memmap",
                              "reward.memmap"}, sorted(files)

        rgb = os.stat(files["obs-rgb.memmap"])
        assert rgb.st_size == CAPACITY * HORIZON * math.prod(IMG), rgb.st_size
        # One of eight slots written, so the file is reserved at full size but allocated
        # at roughly an eighth of it — this is what keeps an unfilled buffer cheap.
        assert rgb.st_blocks * 512 < rgb.st_size / 2, (rgb.st_blocks * 512, rgb.st_size)

        # The storage itself is never copied into RAM...
        assert buf.data["obs", "rgb"].untyped_storage().nbytes() == rgb.st_size
        # ...but indexing it hands back an ordinary detached tensor, not a live view of
        # the slot, so a later wrap cannot mutate a batch already handed to the caller.
        batch = buf.data[torch.tensor([0])]
        assert type(batch["obs", "rgb"]) is torch.Tensor, type(batch["obs", "rgb"])
        buf.add_subtrajectory(window(99))  # does not land in slot 0, but prove it anyway
        assert torch.equal(batch["obs", "rgb"][0], window(0)["obs", "rgb"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_files_are_removed_with_the_buffer():
    """Dropping the buffer takes its scratch directory with it."""
    root = tempfile.mkdtemp(prefix="s2p_test_memmap_")
    try:
        buf = make(root)
        buf.add_subtrajectory(window(0))
        path = buf._storage_path
        assert os.path.isdir(path)

        del buf
        gc.collect()
        assert not os.path.exists(path), path
        assert os.path.isdir(root), "only the buffer's own subdirectory should be removed"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rejects_non_cpu_load_device():
    """A file-backed buffer cannot be allocated on an accelerator."""
    try:
        OnlineTransitionDataset(capacity=1, horizon=1, load_device="cuda", storage_dir="/tmp")
    except AssertionError as e:
        assert "load_device" in str(e), e
        return
    raise AssertionError("expected an AssertionError for load_device='cuda'")


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
