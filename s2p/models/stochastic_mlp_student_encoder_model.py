import torch
import torch.nn as nn
import numpy as np
import typing

from torch.distributions import Normal, Independent

from s2p.lib.stochastic_models import StochasticMLP

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.ema_target_encoder_base import EMATargetEncoderBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference
 
class StochasticMLPStudentEncoderModel(EMATargetEncoderBase, torch.nn.Module):
    def __init__(self, cfg, task:TaskBase, teacher:EncoderModelBase):
        super().__init__(cfg)
        self.cfg = cfg

        self.task = task

        teacher_input_shape, teacher_encoder = teacher.get_encoding_function()
        teacher_output = teacher_encoder(torch.zeros(size=teacher_input_shape, device=self.cfg.device))
        teacher_output_dim = teacher_output.shape[-1]
        
        self.encoder = StochasticMLP(
            input_dim=teacher_output_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.shared_hidden_layers,
            output_dim=cfg.latent_dim,
            head_hidden_dim=cfg.head_hidden_dim,
            head_hidden_layers=cfg.separate_hidden_layers,
            device=cfg.device
        )

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
        distribution = self.encode_distribution(observation)
        # Not gated on `self.training`: the planner runs inside online data collection,
        # which happens under `model.train()`, so an eval-only gate would leave the
        # collection planner sampling while the evaluation planner used means. Losses
        # that need a sample take it from the `*_distribution` method themselves.
        if self.cfg.sample_mean:
            return distribution.mean
        return distribution.rsample()
    
    def encode_distribution(self, observation:torch.Tensor):
        mean, std = self.encoder(observation)
        return Independent(Normal(mean, std), 1)
    
    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        return ((self.task.latent_dim,), self.encoder)
    
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