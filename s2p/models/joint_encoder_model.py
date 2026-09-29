import typing

from omegaconf import OmegaConf
import torch

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference


class _ConcatEncoders(torch.nn.Module):
    """
    Module whose forward pass repeats JointEncoderModel.encode's combination step, for FLOPs
    counting on a single input tensor. Only the first encoder (the one whose shape the
    caller built `x` from) sees the real input; every other encoder gets a zero tensor of its
    own declared shape instead -- shapes are all FLOPs accounting needs, the same trick
    `_DINOWMEncoderForFLOPS` uses for its proprio/action inputs (see dino_world_model.py).
    """
    def __init__(
            self,
            shapes_and_functions: typing.List[typing.Tuple[typing.Tuple[int], torch.nn.Module]],
            combine_latents: typing.Callable,
    ):
        super().__init__()
        self.shapes = [shape for shape, _ in shapes_and_functions]
        self.encoding_functions = torch.nn.ModuleList([fn for _, fn in shapes_and_functions])
        self.combine_latents = combine_latents

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[:x.dim() - len(self.shapes[0])]

        latents = [self.encoding_functions[0](x)]
        for shape, fn in list(zip(self.shapes, self.encoding_functions))[1:]:
            placeholder = torch.zeros(*batch_dims, *shape, device=x.device, dtype=x.dtype)
            latents.append(fn(placeholder))

        return self.combine_latents(latents)


class JointEncoderModel(EncoderModelBase, torch.nn.Module):
    def __init__(
            self,
            cfg: OmegaConf,
            encoders: typing.List[EncoderModelBase],
            task: TaskBase
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.task = task
        self.encoders = torch.nn.ModuleList(encoders)

        if isinstance(self.task.observation_dimension, dict):
            example_obs = {
                key: torch.zeros(
                    size=self.task.observation_dimension[key], 
                    device=self.cfg.device).unsqueeze(0).unsqueeze(0) 
                for key in self.task.observation_dimension
            }
        else:
            example_obs = torch.zeros(
                size=self.task.observation_dimension, 
                device=self.cfg.device
            ).unsqueeze(0).unsqueeze(0)

        concat_dim = sum(enc.encode(example_obs).shape[-1] for enc in encoders)
        self.projection = torch.nn.Linear(concat_dim, self.cfg.downprojection_dim)

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def combine_latents(self, latent_list):
        # Process from fewest dims to most, so lower-dim latents get broadcast
        # into higher-dim ones progressively.
        latents = sorted(latent_list, key=lambda x: x.dim())
        combined = latents[0]

        for nxt in latents[1:]:
            # Insert singleton dims (just before the feature dim) until combined
            # has the same rank as nxt.
            while combined.dim() < nxt.dim():
                combined = combined.unsqueeze(-2)

            # Broadcast combined's spatial dims to match nxt's spatial dims.
            expand_shape = list(nxt.shape[:-1]) + [combined.shape[-1]]
            combined = combined.expand(*expand_shape)

            # Concat features.
            combined = torch.cat([combined, nxt], dim=-1)

        return combined

    def encode(self, observation: torch.Tensor, previous_state: torch.Tensor = None, action: torch.Tensor = None) -> torch.Tensor:
        
        latent_list = [enc.encode(observation, previous_state, action) for enc in self.encoders]
        
        latents = self.combine_latents(latent_list)
        if self.cfg.project: 
            latents = self.projection(latents)
        return latents

    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        shapes_and_functions = [enc.get_encoding_function() for enc in self.encoders]
        input_dim = shapes_and_functions[0][0]
        concat = _ConcatEncoders(shapes_and_functions, self.combine_latents)
        if self.cfg.project:
            return input_dim, torch.nn.Sequential(concat, self.projection)
        else:
            return input_dim, torch.nn.Sequential(concat)

    def save_to_file(self, filepath) -> None:
        # Frozen: these weights are still exactly the checkpoint this was built from, so
        # point at it instead of copying it. See s2p.lib.checkpointing.
        if self.cfg.freeze and self.cfg.checkpoint is not None:
            save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
            return

        torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath) -> None:
        # May be a reference written by `save_to_file` above rather than weights.
        filepath = resolve_checkpoint_path(filepath)
        self.load_state_dict(torch.load(filepath, map_location=self.cfg.device))

    def requires_grad_(self, requires_grad):
        for enc in self.encoders:
            enc.requires_grad_(requires_grad)
