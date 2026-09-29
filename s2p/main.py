import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from s2p.lib.seeding import seed_all
from s2p.runners.runner_base import RunnerBase
from s2p.tasks.base.task_base import TaskBase

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')

    # Before `instantiate`, not inside the runner: Hydra builds the runner's dependencies -- the
    # task, the agent, every evaluator -- before calling `RunnerBase.__init__`, so seeding there
    # happens after all model weights have already been drawn. See s2p/lib/seeding.py.
    seed_all(cfg.seed)

    # Instantiate the Runner
    runner:RunnerBase = instantiate(cfg.runner)

    runner.log_cfg(cfg)

    # Execute it
    runner.run()


if __name__ == "__main__":
    main()