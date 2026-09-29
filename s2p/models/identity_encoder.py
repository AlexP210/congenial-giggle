import typing

import torch

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.tasks.base.task_base import TaskBase

class IdentityEncoder(EncoderModelBase, torch.nn.Module):
    def __init__(self, cfg, task: TaskBase):
        super().__init__(cfg)
        self.cfg = cfg
        self.task = task

        if self.cfg.observation_key is not None:
            self.obs_dim = self.task.observation_dimension[self.cfg.observation_key]
        else:
            self.obs_dim = self.task.observation_dimension

    def encode(self, observation, previous_state: torch.Tensor = None, action: torch.Tensor = None) -> torch.Tensor:
        return observation[self.cfg.observation_key]

    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        return (self.obs_dim, torch.nn.Identity())

    def save_to_file(self, filepath:str) -> None:
        torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath:str) -> None:
        self.load_state_dict(torch.load(filepath, map_location="cpu"))

    def requires_grad_(self, requires_grad):
        return