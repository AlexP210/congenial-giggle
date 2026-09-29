import typing

import torch

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference

class LinearEncoder(EncoderModelBase, torch.nn.Module):
    def __init__(self, cfg, task:TaskBase):
        super().__init__(cfg)
        self.cfg = cfg

        self.task = task
    
        # Get the correct observation dimension
        if self.cfg.observation_keys is not None:
            self.obs_dim = self.task.observation_dimension
            for key in self.cfg.observation_keys:
                self.obs_dim = self.obs_dim[key]
        elif self.cfg.observation_keys is None:
            self.obs_dim = self.task.observation_dimension
    
        self.encoder = torch.nn.Linear(
            in_features=self.obs_dim[-1], # Assume observation tensor ndim=1 for LinearEncoder
            out_features=cfg.latent_dim,
            device=cfg.device
        )

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)


    def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
        
        if self.cfg.observation_key is not None:
            obs = observation[self.cfg.observation_key]
        else:
            obs = observation
        
        return self.encoder(obs)
    
    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        return ((self.obs_dim[-1],), self.encoder)
    
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
