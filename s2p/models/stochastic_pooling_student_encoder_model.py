import typing

import numpy as np
import torch

from torch.distributions import Normal, Independent

from s2p.lib.stochastic_models import StochasticPooling
from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.ema_target_encoder_base import EMATargetEncoderBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference


class StochasticPoolingStudentEncoderModel(EMATargetEncoderBase, torch.nn.Module):
    """
    Student encoder that distils the teacher's token sequence by passing each token
    independently through a shared MLP that maps *that token* to a diagonal Gaussian over
    the latent, then pooling those per-token distributions into a single Gaussian.

    Same interface as StochasticCAPStudentEncoderModel, but with encode-then-pool in place
    of cross-attention pooling, and with the stochasticity produced per token rather than
    after pooling. The token MLP, the token weighting (`pooling`) and the rule combining
    the token distributions (`aggregation`) are configurable.
    """

    def __init__(self, cfg, task: TaskBase, teacher: EncoderModelBase):
        super().__init__(cfg)
        self.cfg = cfg
        self.task = task

        if isinstance(self.task.observation_dimension, dict):
            example_obs = {key: torch.zeros(size=self.task.observation_dimension[key], device=self.cfg.device) for key in self.task.observation_dimension}
        else:
            example_obs = torch.zeros(size=self.task.observation_dimension, device=self.cfg.device)
        teacher_output = teacher.encode(example_obs)
        self.teacher_output_dim = teacher_output.shape
        self.token_dim = self.teacher_output_dim[-1]
        self.sequence_dims = self.teacher_output_dim[:-1]
        self.n_tokens = int(np.prod(self.sequence_dims)) if len(self.sequence_dims) > 0 else 1

        self.encoder = StochasticPooling(
            token_dim=self.token_dim,
            output_dim=cfg.latent_dim,
            token_hidden_dim=cfg.token_hidden_dim,
            token_hidden_layers=cfg.token_hidden_layers,
            token_output_dim=cfg.token_output_dim,
            pooling=cfg.pooling,
            aggregation=cfg.aggregation,
            head_hidden_dim=cfg.head_hidden_dim,
            head_hidden_layers=cfg.head_hidden_layers,
            device=cfg.device,
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

    def encode_distribution(self, observation: torch.Tensor) -> torch.distributions.Distribution:
        # Trailing dims are the teacher's output shape; anything in front is batch (e.g. (T, B)).
        batch_dims = observation.shape[:-len(self.teacher_output_dim)]
        tokens = observation.reshape(-1, self.n_tokens, self.token_dim)  # (T*B, N, token_dim)
        mean, std = self.encoder(tokens)                                 # (T*B, latent_dim)
        mean = mean.reshape(*batch_dims, self.cfg.latent_dim)            # (T, B, latent_dim)
        std = std.reshape(*batch_dims, self.cfg.latent_dim)              # (T, B, latent_dim)
        return Independent(Normal(mean, std), 1)

    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        return ((self.cfg.latent_dim,), self.encoder)

    def requires_grad_(self, requires_grad):
        return super().requires_grad_(requires_grad and not self.cfg.freeze)

    def save_to_file(self, filepath: str) -> None:
        # Frozen: these weights are still exactly the checkpoint this was built from, so
        # point at it instead of copying it. See s2p.lib.checkpointing.
        if self.cfg.freeze and self.cfg.checkpoint is not None:
            save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
            return

        torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath: str) -> None:
        # May be a reference written by `save_to_file` above rather than weights.
        filepath = resolve_checkpoint_path(filepath)
        self.load_state_dict(torch.load(filepath, map_location=self.cfg.device))
