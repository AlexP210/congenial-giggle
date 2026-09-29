import torch
import torch.nn as nn
import numpy as np
import typing

from torch.distributions import Normal, Independent

from s2p.lib.stochastic_models import StochasticMLP

from s2p.models.base.reward_model_base import RewardModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference
 
class StochasticMLPRewardModel(RewardModelBase, torch.nn.Module):
    def __init__(self, cfg, task:TaskBase):
        super().__init__(cfg)
        self.cfg = cfg

        self.task = task
    
        self.reward_model = StochasticMLP(
            input_dim=self.cfg.latent_dim+self.task.action_dimension[-1], # Assume action tensor ndim=1 for MLP
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            output_dim=1,
            head_hidden_dim=cfg.head_hidden_dim,
            head_hidden_layers=cfg.head_hidden_layers,
            device=cfg.device
        )

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def reward(self, s:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
        distribution = self.reward_distribution(s, a)
        # Not gated on `self.training`: the planner runs inside online data collection,
        # which happens under `model.train()`, so an eval-only gate would leave the
        # collection planner sampling while the evaluation planner used means. Losses
        # that need a sample take it from the `*_distribution` method themselves.
        if self.cfg.sample_mean:
            return distribution.mean
        return distribution.rsample()
    
    def reward_distribution(self, s:torch.Tensor, a:torch.Tensor) -> torch.distributions.Distribution:
        s_and_a = torch.cat([s, a], dim=-1)
        mean, std = self.reward_model(s_and_a)
        return Independent(Normal(mean, std), 1)
    
    def get_reward_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        return ((self.cfg.latent_dim+self.task.action_dimension[-1],), self.reward_model)

    def requires_grad_(self, requires_grad):
        return super().requires_grad_(requires_grad and not self.cfg.freeze)

    def save_to_file(self, filepath:str) -> None:
        # Frozen: these weights are still exactly the checkpoint this was built from, so
        # point at it instead of copying it. See s2p.lib.checkpointing.
        if self.cfg.freeze and self.cfg.checkpoint is not None:
            save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
            return

        torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath:str) -> None:
        # May be a reference written by `save_to_file` above rather than weights.
        filepath = resolve_checkpoint_path(filepath)
        self.load_state_dict(torch.load(filepath, map_location=self.cfg.device))