import abc
import typing

from omegaconf import OmegaConf
import torch
import numpy as np

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference

class TeacherStudentEncoderModel(EncoderModelBase, torch.nn.Module):
    def __init__(
            self, 
            cfg:OmegaConf, 
            student:EncoderModelBase,
            teacher:EncoderModelBase
    ):

        super().__init__(cfg)
        self.cfg = cfg

        self.teacher_encoder = teacher
        self.student_encoder = student

        # Only the stochastic students have a distribution to expose. Without this, callers
        # probing for `encode_distribution` on the wrapper would find it and then fail inside
        # on a deterministic student; `None` lets them fall back to `encode` instead.
        if not hasattr(self.student_encoder, "encode_distribution"):
            self.encode_distribution = None

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
        return self.student_encoder.encode(self.teacher_encoder.encode(observation))

    def encode_distribution(self, observation:torch.Tensor, previous_state:torch.Tensor=None, action:torch.Tensor=None):
        """
        The student's latent distribution rather than a sample from it. `encode` draws an
        `rsample`, so it returns a different latent every call for the same observation;
        anything that needs a deterministic latent should take the mean of this instead.
        """
        return self.student_encoder.encode_distribution(self.teacher_encoder.encode(observation))

    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        teacher_encoding_input_dim, teacher_encoding_function = self.teacher_encoder.get_encoding_function()
        student_encoding_input_dims, student_encoding_function = self.student_encoder.get_encoding_function()
        return (
            teacher_encoding_input_dim, 
            torch.nn.Sequential(teacher_encoding_function, student_encoding_function)
        )        
        
    def save_to_file(self, filepath) -> None:
        # Frozen: these weights are still exactly the checkpoint this was built from, so
        # point at it instead of copying it. See s2p.lib.checkpointing.
        if self.cfg.freeze and self.cfg.checkpoint is not None:
            save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
            return

        # If the Teacher is frozen, save only the student
        # The teacher may be massive and there's no need to save
        # the whole thing if it was frozen.
        if self.teacher_encoder.cfg.freeze:
            self.student_encoder.save_to_file(filepath)
        # If the Teacher is not frozen, save the whole module
        # We trained it, and we need the checkpoint
        else:
            torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath) -> None:
        # May be a reference written by `save_to_file` above rather than weights.
        filepath = resolve_checkpoint_path(filepath)
        # If the Teacher is frozem, load only the student -> Teacher should be loaded from its own checkpoint
        if self.teacher_encoder.cfg.freeze:
            self.student_encoder.load_from_file(filepath)
        # Else, we are loading the Teacher too
        else:
            self.load_state_dict(torch.load(filepath, map_location=self.cfg.device))

    def requires_grad_(self, requires_grad):
        self.student_encoder.requires_grad_(requires_grad and not self.cfg.freeze)
        self.teacher_encoder.requires_grad_(requires_grad and not self.cfg.freeze)