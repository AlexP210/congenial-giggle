"""Frame-stack deduplication end to end, against real windows from the online trainer.

Run with (from the project root, with a machines/*.env sourced):

    python agents/squeeze2plan/tests/test_online_buffer_packing_e2e.py [config_name ...]

Defaults to the two online configs on this branch that enable packing. Unlike
`test_online_buffer_packing.py`, which builds windows by hand, this composes the real
config, builds the real ManiSkill env and agent, and collects real frame-stacked
observations (with cached DINO features) through `OnlineTrainer.generate_data`.

The claim under test is that packing is invisible: a packed buffer and a dense one, fed the
same windows, hand back bit-identical batches. So every window the trainer produces is
teed into a second, deliberately unpacked `OnlineTransitionDataset` and the two are
compared entry by entry.
"""
import faulthandler
import os
import signal
import sys
import tempfile

import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from s2p.lib.transition_data import OnlineTransitionDataset

CONFIGS = ["train_visual_online_stochastic", "train_visual_online_deterministic"]
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s2p", "configs")
WINDOWS = 12          # subtrajectories to collect; enough to wrap the buffer below
CAPACITY = 8          # small, so the ring pointer wraps during the test
# cuda:0 rather than the spare GPU: the ManiSkill tasks here pin the simulator with an
# indexed `task.cfg.sim_backend` (physx_cuda:0), and putting torch on a different device
# than sapien is its own class of failure. Override with S2P_TEST_DEVICE if both move.
DEVICE = os.environ.get("S2P_TEST_DEVICE", "cuda:0")


def identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-identical, counting NaN as equal to NaN.

    `torch.equal` will not do: the entry belonging to a reset observation has no action or
    reward that produced it, so `OnlineTrainer.generate_subtrajectory` fills both with NaN
    (the losses drop index 0). Every window therefore carries NaNs at t=0, and `torch.equal`
    would report every one of them as a difference.
    """
    if not a.shape == b.shape or a.dtype != b.dtype:
        return False
    if not a.is_floating_point():
        return bool(torch.equal(a, b))
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


def build(config_name, output_dir):
    """Instantiate the runner from `config_name`, with only the run's plumbing overridden."""
    overrides = [
        f"data_dir={os.environ['DATA_DIR']}",
        f"checkpoint_dir={os.environ['CHECKPOINT_DIR']}",
        f"output_dir={output_dir}",
        f"device={DEVICE}",
        # Evaluators each build their own env and are not what is under test here.
        "runner.evaluators=null",
        "runner.cfg.use_wandb=false",
        "runner.cfg.compile=false",
        f"runner.cfg.run_name=packing_e2e_{config_name}",
        f"trainer.cfg.buffer_size={CAPACITY}",
        # Act randomly: the planner is irrelevant to storage and costs ~300 ms a step.
        f"trainer.cfg.seed_steps={10 ** 9}",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        # `return_hydra_config` + `set_config` because the vendored TD-MPC2 package calls
        # `hydra.utils.get_original_cwd()` while its value model is built, and that reads
        # the HydraConfig singleton, which `compose` leaves unset (only `@hydra.main` fills
        # it in). The `hydra` node is dropped afterwards so `cfg` is the run config again.
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    OmegaConf.set_struct(cfg, False)
    del cfg["hydra"]
    OmegaConf.set_struct(cfg, True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    return instantiate(cfg.runner), cfg


def check(config_name):
    with tempfile.TemporaryDirectory() as output_dir:
        print("    building runner...", flush=True)
        runner, cfg = build(config_name, output_dir)
        print("    runner built", flush=True)
        trainer, model = runner.trainer, runner.model
        packed = trainer.dataset

        num_stacked_frames = cfg.trainer.cfg.num_stacked_frames
        horizon = int(cfg.trainer.cfg.horizon)
        assert num_stacked_frames == cfg.task.cfg.num_frames, (
            f"{config_name}: trainer num_stacked_frames ({num_stacked_frames}) disagrees with "
            f"task num_frames ({cfg.task.cfg.num_frames})"
        )

        # The control: same windows, stored the old dense way.
        dense = OnlineTransitionDataset(
            capacity=CAPACITY, horizon=horizon,
            load_device=cfg.trainer.cfg.load_device,
            batch_device=cfg.trainer.cfg.batch_device,
            num_stacked_frames=None,
        )
        # Tee every window the trainer produces into the control buffer, and count them:
        # `len(dataset)` saturates at capacity, so it cannot say how many were collected.
        # The trainer appends a `[horizon, num_envs]` batch per collection call, so the tee goes
        # on `add_subtrajectories` and unpacks it the same way the buffer does -- which also keeps
        # the control buffer's entry order identical to the packed one's.
        collected = []
        add_to_packed = packed.add_subtrajectories

        def tee(td):
            for env_index in range(td.batch_size[1]):
                dense.add_subtrajectory(td[:, env_index].clone())
                collected.append(None)
            return add_to_packed(td)

        packed.add_subtrajectories = tee

        model.eval()
        model.requires_grad_(False)
        print("    collecting...", flush=True)
        while len(collected) < WINDOWS:
            trainer.generate_data(model)

        assert len(collected) >= WINDOWS > CAPACITY, (len(collected), WINDOWS, CAPACITY)
        assert len(packed) == len(dense) == CAPACITY, (len(packed), len(dense))
        assert packed.packed_keys, f"{config_name}: nothing was packed -- is num_stacked_frames set?"

        # Every entry, including the ones the ring pointer overwrote.
        for i in range(len(packed)):
            got, want = packed[i], dense[i]
            for key in dense.keys:
                assert identical(got[key], want[key]), f"{config_name}: entry {i}, key {key}"

        # And the same through the sampling path the trainer actually uses.
        torch.manual_seed(0)
        got_batch = packed.sample(4)
        torch.manual_seed(0)
        want_batch = dense.sample(4)
        assert got_batch.batch_size == want_batch.batch_size, (got_batch.batch_size, want_batch.batch_size)
        for key in dense.keys:
            assert identical(got_batch[key], want_batch[key]), f"{config_name}: sample, key {key}"

        saved = sum(dense.data[k].numel() * dense.data[k].element_size() for k in dense.keys)
        cost = sum(packed.data[k].numel() * packed.data[k].element_size() for k in packed.keys)
        print(f"  {config_name}: horizon={horizon} num_stacked_frames={num_stacked_frames}")
        print(f"    packed keys: {sorted(str(k) for k in packed.packed_keys)}")
        print(f"    {len(packed)} entries bit-identical to the dense buffer, sample() too")
        print(f"    buffer bytes {saved / 2**20:.1f} MiB dense -> {cost / 2**20:.1f} MiB packed "
              f"({saved / cost:.2f}x)")

        del runner, trainer, model, packed, dense
        torch.cuda.empty_cache()


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)   # `kill -USR1 <pid>` dumps where it is
    for name in (sys.argv[1:] or CONFIGS):
        print(f"[{name}]")
        check(name)
    print("\nall end-to-end packing checks passed")
