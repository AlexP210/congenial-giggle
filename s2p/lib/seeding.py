"""One place to seed every global RNG a run draws from.

Why this is a module rather than a few lines in the runner: Hydra instantiates a config's
dependencies *before* the object that declares them, so `instantiate(cfg.runner)` builds the task,
the agent and every evaluator -- including all model weight initialization -- before
`RunnerBase.__init__` ever executes. Seeding inside the runner therefore happens strictly after
the weights exist, and two runs of the same config get different networks.

So `seed_all` is called from `main.py` before instantiation, which is the only point that precedes
every consumer. `RunnerBase` deliberately does *not* call it: seeding there is too late to affect
initialization, and doing it anyway would make an unseeded run look seeded, which is what hid this
for so long. Anything that builds S2P components outside `main.py` and wants reproducibility has
to call this itself, before `instantiate`.
"""
import os
import random

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices).

    Args:
        seed: the run's seed.
        deterministic: also pin cuDNN to deterministic algorithms and disable its autotuner.
            Costs some throughput, and is what makes two runs of the same config agree
            step for step rather than only in distribution.
    """
    seed = int(seed)
    # PYTHONHASHSEED only takes effect for subprocesses started after this point; set-iteration
    # order in *this* process was already fixed at startup. Harmless, and it covers workers.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_spaces(*spaces) -> None:
    """Seed gymnasium `Space` RNGs, which `seed_all` cannot reach.

    `Space.sample()` draws from the space's own `np_random`, seeded from OS entropy when the space
    is constructed -- not from the global numpy RNG. That is what `rand_act()` uses to fill the
    replay buffer during the seed-step phase, so leaving it alone makes the initial data, and
    therefore every run, different even when the weights match.

    Each space's seed is drawn from the global numpy RNG rather than taken as an argument: callers
    then need no `seed` plumbed through their config, and spaces created in sequence get different
    -- but reproducible -- streams instead of all replaying the same actions. Only deterministic
    if `seed_all` ran first, which is the point.
    """
    for space in spaces:
        if space is not None:
            space.seed(int(np.random.randint(0, 2 ** 31 - 1)))
