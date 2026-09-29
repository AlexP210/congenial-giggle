import math
import typing

import torch

from s2p.lib.deterministic_models import DeterministicCAP
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference


class DINOWMRewardModel(RewardModelBase, torch.nn.Module):
    """
    A cross-attention reward head over a DINO-WM-family latent window.

    Works with any of the four wrappers -- `DINOWorldModel`, `TCWorldModel`,
    `DINOBisimWorldModel`, `SparseImaginationWorldModel` -- because it asks the encoder for
    only two things: `_unflatten_latent`, to get the `(batch, num_frames, tokens, dim)` window
    back out of a flat state, and `write_action_into_newest_frame`, to condition on an action
    the way that wrapper's own predictor does. Whatever the window's token count and width
    turn out to be is read off a probe encode in `__init__`, so a 64-wide bisimulation token
    and a 404-wide DINO-WM one both work without a config change.

    Deliberately no `isinstance` check and no import of any wrapper: the four repos claim the
    same rootless top-level module names, so importing one to type-check against it would make
    the other three unimportable in the same process (see `s2p.lib.tc_wm_path`).
    """

    def __init__(self, cfg, task: TaskBase, encoder):
        super().__init__(cfg)
        self.cfg = cfg
        self.task = task
        self.encoder = encoder

        # teacher_input_shape, teacher_encoder = teacher.get_encoding_function()
        if isinstance(self.task.observation_dimension, dict):
            example_obs = {key: torch.zeros(size=self.task.observation_dimension[key], device=self.cfg.device) for key in self.task.observation_dimension}
        else:
            example_obs = torch.zeros(size=self.task.observation_dimension, device=self.cfg.device)
        dino_wm_encoder_output = encoder._unflatten_latent(encoder.encode(example_obs))

        # `_unflatten_latent` collapses whatever batch dims it is given into a single leading
        # axis, so probing it with one unbatched observation comes back as
        # `(1, num_hist, tokens, dim)`. The leading 1 is that collapsed axis, not part of the
        # latent — what the reward head needs is the per-sample shape below it.
        self.latent_shape = dino_wm_encoder_output.shape[1:]   # (num_hist, tokens, dim)
        self.token_dim = self.latent_shape[-1]
        self.n_tokens = math.prod(self.latent_shape[:-1])       # num_hist * tokens

        self.reward_model = DeterministicCAP(
            token_dim=self.token_dim,
            mha_embedding_dim=cfg.embedding_dim,
            mha_n_heads=cfg.n_heads,
            output_dim=1,
            mha_n_queries=cfg.n_queries,
            ff_hidden_layers=cfg.ff_hidden_layers,
            ff_hidden_dim=cfg.ff_hidden_dim,
            head_hidden_layers=cfg.head_hidden_layers,
            head_hidden_dim=cfg.head_hidden_dim,
            n_blocks=cfg.n_blocks,
            device=cfg.device,
        )

        if self.cfg.checkpoint is not None:
            self.load_from_file(self.cfg.checkpoint)

    def reward(self, state:torch.Tensor, action:torch.Tensor=None) -> torch.Tensor:
        # The batch dims are read off `state`, not off the unflattened latent: the latter has
        # already had them collapsed into its leading axis, so they are unrecoverable there.
        batch_dims = state.shape[:-1]
        z = self.encoder._unflatten_latent(state)  # (prod(batch_dims), num_hist, tokens, dim)

        if action is not None:
            # Condition on the action the way the world model itself does: write it into the
            # newest frame's action channels, which is the first of the three steps the
            # wrapper's `dynamics` runs before predicting. A latent that reaches here has
            # those channels cleared — `encode` fills them with the zero-action embedding and
            # `dynamics` clears the frame it just predicted — so this is writing into an empty
            # slot rather than over a real action either way.
            #
            # The wrapper's own method rather than its inner model's, both so that each
            # repo's action normalization and `concat_dim` layout come from the one place that
            # knows them, and because it copies: `replace_actions_from_z` writes in place, and
            # MPPI reuses the latent it passes in (it scores this reward, then rolls the same
            # `z` forward).
            z = self.encoder.write_action_into_newest_frame(z, action)

        # One frame's tokens and the next frame's are pooled by the same cross-attention, so
        # the whole predictor window becomes one flat token sequence.
        tokens = z.reshape(-1, self.n_tokens, self.token_dim)
        return self.reward_model(tokens).reshape(*batch_dims, 1)

    def get_reward_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
        # Same contract as every DINO-WM-family wrapper's own `get_reward_function`: a flat
        # `(latent, action)` input, of the widths `reward` itself takes. The action width is the
        # *task's*, not the world model's — `reward` is handed a task action and leaves the
        # tiling over `frameskip` to the wrapper's `_to_dinowm_action`.
        reward_module = _DINOWMRewardHeadForFLOPS(
            encoder=self.encoder,
            reward_model=self.reward_model,
            latent_dim=self.n_tokens * self.token_dim,
            action_dim=self.task.action_dimension[-1],
            n_tokens=self.n_tokens,
            token_dim=self.token_dim,
        )
        return (
            (self.n_tokens * self.token_dim + self.task.action_dimension[-1],),
            reward_module,
        )

    def requires_grad_(self, requires_grad):
        return super().requires_grad_(requires_grad and not self.cfg.freeze)

    def save_to_file(self, filepath: str) -> None:
        # Frozen: these weights are still exactly the checkpoint this was built from, so
        # point at it instead of copying it. See s2p.lib.checkpointing.
        if self.cfg.freeze and self.cfg.checkpoint is not None:
            save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
            return

        # `self.encoder` is the DINO-WM this head reads latents from, and it is a submodule,
        # so a plain `self.state_dict()` drags the whole world model into every checkpoint —
        # ~205 MiB of frozen weights next to a ~13 MiB head. Same reasoning, and the same
        # split, as TeacherStudentEncoderModel: a frozen encoder is loaded from its own
        # checkpoint, so only the trained part belongs here.
        if self.encoder.cfg.freeze:
            torch.save(self.reward_model.state_dict(), filepath)
        else:
            torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath: str) -> None:
        # May be a reference written by `save_to_file` above rather than weights.
        filepath = resolve_checkpoint_path(filepath)
        if self.encoder.cfg.freeze:
            self.reward_model.load_state_dict(torch.load(filepath, map_location=self.cfg.device))
        else:
            self.load_state_dict(torch.load(filepath, map_location=self.cfg.device))


class _DINOWMRewardHeadForFLOPS(torch.nn.Module):
    """
    Module whose forward pass repeats `DINOWMRewardModel.reward` from a flat (latent, action)
    input. Used for counting the FLOPs of the reward head only.

    The whole path is replayed, action write included: it is a clone of the newest frame plus a
    concat over the rest of the window, which for a 238k-float DINO-WM latent is real work per
    MPPI sample and belongs in the timing `FLOPSEvaluator` takes alongside the FLOP count.

    The encoder is deliberately held *outside* the module tree (`object.__setattr__`, so
    `nn.Module.__setattr__` does not register it). It is the world model this head reads latents
    from, and registering it would put its ~22M frozen parameters — a DINOv3 backbone and a ViT
    predictor, neither of which this path runs — into the reward head's reported parameter
    count. Same split, and the same reasoning, as `DINOWMRewardModel.save_to_file`. The cost of
    that exclusion is that the action encoder's 1x1 convolution goes unhooked and so uncounted;
    it is a convolution over a single ~4-dim action, against a cross-attention pooling of
    `num_hist * tokens` tokens.
    """

    def __init__(self, encoder, reward_model, latent_dim, action_dim, n_tokens, token_dim):
        super().__init__()
        object.__setattr__(self, "encoder", encoder)
        self.reward_model = reward_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim

    def forward(self, latent_action: torch.Tensor):
        latent, action = latent_action.split((self.latent_dim, self.action_dim), dim=-1)
        z = self.encoder._unflatten_latent(latent)
        z = self.encoder.write_action_into_newest_frame(z, action)
        tokens = z.reshape(-1, self.n_tokens, self.token_dim)
        return self.reward_model(tokens)
