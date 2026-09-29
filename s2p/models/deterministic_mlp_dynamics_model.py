import torch
import torch.nn as nn
import numpy as np
import typing

from torch.distributions import Normal, Independent

from s2p.lib.deterministic_models import DeterministicMLP

from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference
 
class DeterministicMLPDynamicsModel(DynamicsModelBase, torch.nn.Module):
    def __init__(self, cfg, task:TaskBase):
        super().__init__(cfg)
        self.cfg = cfg

        self.task = task
    
        self.dynamics_model = DeterministicMLP(
            input_dim=self.cfg.latent_dim+self.task.action_dimension[-1], # Assume action tensor ndim=1 for MLP
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers,
            output_dim=self.cfg.latent_dim,
            device=cfg.device
        )

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def dynamics(self, s:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
        s_and_a = torch.cat([s, a], dim=-1)
        return self.dynamics_model(s_and_a)
    
    def get_dynamics_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        return ((self.cfg.latent_dim+self.task.action_dimension[-1],), self.dynamics_model)

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