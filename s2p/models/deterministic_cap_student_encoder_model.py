import typing

import torch

from s2p.lib.deterministic_models import DeterministicCAP
from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.ema_target_encoder_base import EMATargetEncoderBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference


class DeterministicCAPStudentEncoderModel(EMATargetEncoderBase, torch.nn.Module):
    def __init__(self, cfg, task: TaskBase, teacher: EncoderModelBase):
        super().__init__(cfg)
        self.cfg = cfg
        self.task = task

        # teacher_input_shape, teacher_encoder = teacher.get_encoding_function()
        if isinstance(self.task.observation_dimension, dict):
            example_obs = {key: torch.zeros(size=self.task.observation_dimension[key], device=self.cfg.device) for key in self.task.observation_dimension}
        else:
            example_obs = torch.zeros(size=self.task.observation_dimension, device=self.cfg.device)
        teacher_output = teacher.encode(example_obs)
        self.teacher_output_dim = teacher_output.shape
        self.token_dim = self.teacher_output_dim[-1]
        self.sequence_dims = self.teacher_output_dim[:-1]

        self.encoder = DeterministicCAP(
            token_dim=self.token_dim,
            mha_embedding_dim=cfg.embedding_dim,
            mha_n_heads=cfg.n_heads,
            output_dim=cfg.latent_dim,
            mha_n_queries=cfg.n_queries,
            ff_hidden_layers=cfg.ff_hidden_layers,
            ff_hidden_dim=cfg.ff_hidden_dim,
            n_blocks=cfg.n_blocks,
            device=cfg.device,
        )

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:

        batch_dims = observation.shape[:-len(self.teacher_output_dim)]
        batch_dims_flattened = observation.flatten(0, len(batch_dims)-1)
        sequence_dims_flattened = batch_dims_flattened.flatten(1, 1+len(self.sequence_dims)-1)
        out = self.encoder(sequence_dims_flattened)  # (T*B, latent_dim)
        return out.unflatten(0, batch_dims)                # (T, B, latent_dim)

    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        encoder_module = _DeterministicCAPForFLOPS(
            encoder=self.encoder,
            teacher_output_dim=self.teacher_output_dim,
            sequence_dims=self.sequence_dims,
        )
        return ((self.cfg.latent_dim,), encoder_module)

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


class _DeterministicCAPForFLOPS(torch.nn.Module):
    """
    Module whose forward pass repeats `encode`'s reshape into (batch, sequence, token_dim)
    ahead of the cross-attention encoder. Without it, the encoder sees the teacher's raw
    multi-axis output (e.g. `(B, S, num_tokens, D)`) instead of the flattened sequence its
    `nn.MultiheadAttention` blocks require. Mirrors `_StochasticCAPForFLOPS`.
    """
    def __init__(self, encoder, teacher_output_dim, sequence_dims):
        super().__init__()
        self.encoder = encoder
        self.teacher_output_dim = teacher_output_dim
        self.sequence_dims = sequence_dims

    def forward(self, observation: torch.Tensor):
        batch_dims = observation.shape[:-len(self.teacher_output_dim)]
        batch_dims_flattened = observation.flatten(0, len(batch_dims)-1)
        sequence_dims_flattened = batch_dims_flattened.flatten(1, 1+len(self.sequence_dims)-1)
        return self.encoder(sequence_dims_flattened).unflatten(0, batch_dims)
