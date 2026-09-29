import contextlib
import typing
import math
import os

import torch
from torchvision.transforms import v2
import torch.nn.functional as F
from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.tasks.base.task_base import TaskBase

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

def make_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

_DINO_OUTPUT_DIMS = {
    'dinov3_vits16':         384,
    'dinov3_vits16plus':     384,
    'dinov3_vitb16':         768,
    'dinov3_vitl16':        1024,
    'dinov3_vitl16plus':    1024,
    'dinov3_vith16plus':    1280,
    'dinov3_vit7b16':       4096,
    'dinov3_convnext_tiny':   768,
    'dinov3_convnext_small':  768,
    'dinov3_convnext_base':  1024,
    'dinov3_convnext_large': 1536,
}


class DINOV3EncoderModel(EncoderModelBase, torch.nn.Module):
    def __init__(self, cfg, task: TaskBase):
        super().__init__(cfg)
        self.cfg = cfg
        self.task = task

        # Get the correct observation dimension
        if self.cfg.observation_key is not None:
            self.obs_dim = self.task.observation_dimension[self.cfg.observation_key]
        elif self.cfg.observation_key is None:
            self.obs_dim = self.task.observation_dimension

        if len(self.obs_dim) != 4 or self.obs_dim[1] != 3:
            raise ValueError(
                "Task observation dim must be (S, 3, *, *) corresponding to frame-stacked (S), "
                "channel-first RGB images. If the target task is state-based, consider using an "
                "MLP-based encoder."
            )

        # `pretrained: False` builds the architecture and calls its own `init_weights()`
        # instead of loading `checkpoint`, which is what a from-scratch encoder ablation
        # needs. `weights` is only read inside the hub's `pretrained` branch, so leaving
        # `checkpoint: null` alongside it is fine. Defaults to True so every existing
        # config keeps loading its checkpoint.
        pretrained = getattr(cfg, 'pretrained', True)
        self.backbone = torch.hub.load(
            os.path.join(PROJECT_ROOT, 'dependencies', 'dinov3'),
            cfg.model_name,
            source="local",
            pretrained=pretrained,
            weights=cfg.checkpoint,
        )
        self.backbone = self.backbone.to(cfg.device)

        self.token_length = _DINO_OUTPUT_DIMS[cfg.model_name]
        if cfg.token_mode == 'patch':
            num_tokens = (cfg.resize_size // 16) ** 2
            self.num_tokens = num_tokens
        elif cfg.token_mode in ('cls', 'pool'):  # 'cls' or 'pool'
            self.num_tokens = 1
        self.transform = make_transform(cfg.resize_size)

        # Set by SaliencyEvaluator (via `model.modules()`) to get gradients with respect to
        # pixels: the `no_grad` below leaves the input with no gradient path at all, so a
        # saliency map would come back empty. Freezing is about not updating these weights,
        # not about refusing to build a graph.
        self.force_pixel_gradients = False

        # if self.cfg.checkpoint is not None:
        #     self.load_from_file(self.cfg.checkpoint)

    def encode(self, observation: torch.Tensor, previous_state: torch.Tensor = None, action: torch.Tensor = None) -> torch.Tensor:
        
        if self.cfg.observation_key is not None:
            obs = observation[self.cfg.observation_key]
        else:
            obs = observation
        
        # observation: (T, B, S, 3, H, W)
        batch_dims = obs.shape[:-len(self.obs_dim)]
        S, C, H, W = obs.shape[-len(self.obs_dim):]

        # (T, B, S, 3, H, W) -> (T*B*S, 3, H, W)
        prod_batch_dims = math.prod(batch_dims)
        x = obs.reshape(prod_batch_dims * S, C, H, W)

        # An unfrozen backbone inherits the caller's grad mode rather than forcing it on: forcing it
        # kept a full autograd graph through the ViT for every no_grad rollout encode.
        if self.force_pixel_gradients:
            ctx = torch.enable_grad()
        elif self.cfg.freeze:
            ctx = torch.no_grad()
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            if self.cfg.token_mode == 'patch':
                feats = self.backbone.forward_features(self.transform(x))
                x = feats['x_norm_patchtokens']  # (T*B*S, N, D)
            elif self.cfg.token_mode == 'pool':
                feats = self.backbone.forward_features(self.transform(x))
                x = feats['x_norm_patchtokens'].mean(dim=1, keepdim=True)  # (T*B*S, 1, D)
            elif self.cfg.token_mode == 'cls':
                x = self.backbone(self.transform(x)).unsqueeze(1)  # (T*B*S, 1, D)

        return x.reshape(*batch_dims, S, self.num_tokens, self.token_length)

    def train(self, mode: bool = True):
        """
        Hold a frozen backbone in eval mode regardless of the mode set on the agent.

        DINOv3 augments its RoPE position encoding while `training` is set - the patch
        coordinates get a random shift, jitter and rescale (see
        `dinov3/layers/rope_position_encoding.py`) - so a backbone left in train mode
        returns a *different* encoding of the same frame on every call. For a teacher
        that is being distilled from, that is noise injected straight into the target,
        and it also puts training-time features in a different space from the ones every
        evaluator sees under `model.eval()`. Freezing the weights should freeze the
        function they compute.
        """
        super().train(mode)
        if self.cfg.freeze:
            self.backbone.eval()
        return self

    def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        encoder_module = _DINOV3EncoderForFLOPS(
            backbone=self.backbone,
            transform=self.transform,
            obs_dim=self.obs_dim,
            token_mode=self.cfg.token_mode,
            num_tokens=self.num_tokens,
            token_length=self.token_length,
        )
        return (self.obs_dim, encoder_module)

    def requires_grad_(self, requires_grad):
        return super().requires_grad_(requires_grad and not self.cfg.freeze)

    def save_to_file(self, filepath: str) -> None:
        # No checkpoint reference here, deliberately. `cfg.checkpoint` is handed to
        # `torch.hub.load(weights=...)` in `__init__`, which is a backbone weights file this
        # class's own `load_from_file` cannot read -- so pointing a checkpoint folder at it
        # would write an entry that does not load back. The weights are written out in full
        # instead. See s2p.lib.checkpointing for the guard every other frozen model uses.
        torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, map_location=self.cfg.device))


class _DINOV3EncoderForFLOPS(torch.nn.Module):
    """
    Module whose forward pass repeats `DINOV3EncoderModel.encode`'s tensor path, without the
    observation-dict indexing `encode` does first. Used for counting the FLOPs of the
    backbone alone, on a bare `(*, *obs_dim)` tensor rather than a full observation.
    """
    def __init__(self, backbone, transform, obs_dim, token_mode, num_tokens, token_length):
        super().__init__()
        self.backbone = backbone
        self.transform = transform
        self.obs_dim = obs_dim
        self.token_mode = token_mode
        self.num_tokens = num_tokens
        self.token_length = token_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[:-len(self.obs_dim)]
        S, C, H, W = x.shape[-len(self.obs_dim):]
        prod_batch_dims = math.prod(batch_dims)
        x = x.reshape(prod_batch_dims * S, C, H, W)

        if self.token_mode == 'patch':
            feats = self.backbone.forward_features(self.transform(x))
            x = feats['x_norm_patchtokens']
        elif self.token_mode == 'pool':
            feats = self.backbone.forward_features(self.transform(x))
            x = feats['x_norm_patchtokens'].mean(dim=1, keepdim=True)
        elif self.token_mode == 'cls':
            x = self.backbone(self.transform(x)).unsqueeze(1)

        return x.reshape(*batch_dims, S, self.num_tokens, self.token_length)
