import typing
import hashlib
import os
import contextlib

import torch

import wandb
from einops import rearrange, repeat
from omegaconf import OmegaConf

from hydra.utils import instantiate

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.lib.sparse_imagination_path import ensure_sparse_imagination_importable
from s2p.lib.checkpointing import resolve_checkpoint_path, resume_folder_checkpoint, save_checkpoint_reference


# Submodules that a native Sparse Imagination checkpoint (Trainer.save_ckpt) stores as whole
# pickled nn.Modules rather than as state dicts, mirroring plan.py's ALL_MODEL_KEYS.
_SPARSE_MODULE_KEYS = ("encoder", "predictor", "decoder", "proprio_encoder", "action_encoder")


def _import_sparse_imagination():
	"""
	Import Sparse Imagination's `Trainer`, world models and image transform, putting its root
	on `sys.path` first.

	Deliberately *not* done at module import, unlike `s2p.models.dino_world_model`'s
	`from dino_wm.train import Trainer`: this repo, DINO-WM and TC-WM all claim the same
	rootless top-level module names and cannot both be importable in one process. Deferring
	keeps merely *importing* this module harmless — see `s2p.lib.sparse_imagination_path`,
	which raises with an explanation when the conflict is real.
	"""
	ensure_sparse_imagination_importable()
	from train import Trainer
	from models.visual_world_model import VWorldModel, VWorldModelDrop
	from dataloaders.img_transforms import default_transform
	return Trainer, VWorldModel, VWorldModelDrop, default_transform


@contextlib.contextmanager
def _wandb_suppressed():
	"""
	Keep Sparse Imagination's Trainer from claiming this process's wandb run.

	`Trainer.__init__` calls `wandb.init(project="wm_robot")` unconditionally and then renames
	the run after the world model. wandb's `reinit` default is "return the previous run", so
	every later init — including the one EvaluationRunner makes into the S2P project — hands
	back that first run, and every S2P metric is logged into `wm_robot` under the world
	model's name instead.

	Here Trainer is only a model factory, so its logging is unwanted outright rather than
	misdirected: `WANDB_MODE=disabled` turns its init into a NoopRun that still satisfies
	everything it does with the handle (`.id`, renaming, `watch`). The mode is cached in
	wandb's process-global setup, though — leaving it there would silently no-op the
	runner's own init too — so `teardown()` clears it on the way out.

	A run that was already active on entry is left alone: `reinit` means Trainer would have
	adopted (and renamed) it whatever this did, and tearing it down would be worse.
	"""
	preexisting_run = wandb.run
	previous_mode = os.environ.get("WANDB_MODE")
	os.environ["WANDB_MODE"] = "disabled"
	try:
		yield
	finally:
		if preexisting_run is None:
			wandb.finish()
		if previous_mode is None:
			os.environ.pop("WANDB_MODE", None)
		else:
			os.environ["WANDB_MODE"] = previous_mode
		if preexisting_run is None:
			wandb.teardown()


@contextlib.contextmanager
def _global_rng_from(generator: torch.Generator):
	"""
	Run the body with `generator`'s stream installed as the global CPU RNG.

	`VWorldModelDrop.reset_random_patches` draws its kept-patch subset with a bare
	`torch.randperm`, i.e. from the global generator, which makes the subset a function of
	however much randomness everything else in the process happened to consume first — and
	silently different between two runs of the same config. Swapping a private, seeded
	generator in for the duration makes the sequence of subsets reproducible without either
	perturbing the global stream or forking the repo's own implementation.

	The private generator is advanced by whatever the body drew, so successive resamples
	continue the same reproducible sequence rather than repeating the first draw.
	"""
	saved = torch.get_rng_state()
	torch.set_rng_state(generator.get_state())
	try:
		yield
	finally:
		generator.set_state(torch.get_rng_state())
		torch.set_rng_state(saved)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
	"""
	Peel DDP / `torch.compile` wrappers off a module.

	Trainer runs every submodule through `accelerator.prepare`, so what it hands back may be
	a DistributedDataParallel (or an OptimizedModule) around the real one. Attribute access
	proxies through those wrappers, but `state_dict` keys do not — they gain a `module.` /
	`_orig_mod.` prefix, which would make every checkpoint written here unloadable by the
	original Sparse Imagination code. Unwrap once, up front, so both stay in the same
	namespace. The repo's own `_unwrap_module` does the same thing one level deep.
	"""
	seen = set()
	while id(module) not in seen:
		seen.add(id(module))
		inner = getattr(module, "_orig_mod", None)
		if inner is None:
			inner = getattr(module, "module", None)
		if not isinstance(inner, torch.nn.Module):
			break
		module = inner
	return module


class SparseImaginationWorldModel(
	EncoderModelBase,
	DynamicsModelBase,
	# PolicyModelBase,       # Sparse Imagination has no policy
	torch.nn.Module
):
	"""
	Sparse Imagination wrapper, exposing a trained `VWorldModel` / `VWorldModelDrop` through
	the encoder and dynamics interfaces the planners and losses in this package expect.

	The counterpart of `s2p.models.dino_world_model.DINOWorldModel`, and structured the same
	way, because Sparse Imagination is a DINO-WM fork: same ViT predictor over a history
	window, same `[visual | proprio | action]` token layout, same frameskip/action-chunk
	convention. Two things differ, and everything below follows from them.

	### Sparse imagination: planning on a subset of the patch tokens

	The paper's claim is that imagination does not need every visual token. Training applies
	random patch dropout (`drop_rate_ub`) with a predictor that has *frame-only* positional
	embeddings (`predictor=vit_nope`, `ViTPredictorWithoutPE`), so the predictor never learns
	to depend on which patch sits where. Planning then keeps a random subset of
	`plan_num_kept_patches` of the encoder's patch tokens and rolls out on those alone —
	`VWorldModelDrop`, selected by `cfg.drop`.

	What that buys is in the *dynamics*, not the encoder: the backbone still runs on the whole
	frame and the subset is taken from its output, so `get_encoding_function` reports the same
	cost either way, while the predictor's attention drops quadratically in the token count.
	For dinov3_vits16 at 224px with `num_hist=3` and a 404-wide token, the latent this class
	hands out goes from 3*196*404 ≈ 238k floats per sample to 3*98*404 ≈ 119k at `drop98` —
	which matters directly for an MPPI population, since the population is `num_samples`
	copies of it.

	The subset is drawn at construction and stays put for the model's lifetime, because it is
	part of what a latent *means*: two latents are only comparable if they were encoded under
	the same one. `resample_kept_patches` draws a new one, but nothing in this package calls
	it — see that method for why the repo's own per-replan
	`MPCPlanner._refresh_random_patches` has no safe equivalent here, and what a caller has to
	re-encode if it does invoke it.

	### What "state" means here

	Sparse Imagination is not Markov in a single frame: its predictor is a ViT over
	`num_hist * tokens_per_frame` tokens, so a one-step prediction needs the whole history
	window. The latent this wrapper hands out is therefore the *flattened window*

		s = flatten(z[:, -num_hist:])        # (num_hist * tokens_per_frame * token_dim,)

	where `z` is exactly the repo's own latent — visual patch tokens (all of them, or the kept
	subset) with the proprio and action embeddings concatenated in (`concat_dim=1`) or
	appended as extra tokens (`concat_dim=0`). `dynamics` writes the incoming action into the
	newest frame's action slot, calls `predictor`, and slides the window by one, which is
	precisely what `VWorldModel.rollout` does.

	Frame `i`'s action slot holds the action applied *at* frame `i`, taking it to frame
	`i+1`, so the newest frame's action is one nothing has chosen yet. Every latent this
	class hands out — from `encode` and from `dynamics` alike — therefore carries zero
	there, and only `dynamics` fills it in, for the duration of one prediction. See
	`_action_slots`.

	### Where the reward comes from

	Nowhere in here. This is an encoder and a dynamics model, and nothing else: it implements
	neither `RewardModelBase` nor `StateValueModelBase`, so `AgentModel` cannot be handed it
	as `reward_model` or `value_model` and `MPPIPlanner` will never ask it to score anything.

	That is a deliberate narrowing of what the repo does. Upstream it plans by driving the
	latent towards a goal latent and scoring a trajectory by where it ends up
	(`create_objective_fn(mode="last")`), which makes the model goal-conditioned: undefined
	until a goal is set, and re-aimed at every reset because the goal belongs to the episode
	rather than to the model.

	Instead the reward is a *separate* model fitted on top of this one:
	`s2p.models.dino_wm_reward_model.DINOWMRewardModel`, a cross-attention head over this
	latent window, trained by `s2p.losses.dino_wm_reward_loss.DINOWMRewardLoss` against the
	task's recorded rewards (`train_sparse_imagination_reward_*.yaml`) and then loaded frozen
	beside this model at evaluation time (`frozen_sparse_imagination_reward_model_*.yaml`).
	The two meet through `_unflatten_latent` and `write_action_into_newest_frame`, which are
	the only things the head asks of a world model — the same two on all four DINO-WM-family
	wrappers, which is why one head serves all of them.

	It also settles the kept-patch subset: the head is fitted against latents built from one
	subset, so the subset has to stay fixed for the head to mean anything. See
	`resample_kept_patches`.

	The practical consequence is that this model plans with an ordinary per-step reward, so
	`planner.cfg.use_value: false` is correct for it and `discount` trades intermediate reward
	against terminal value the way it does anywhere else. It also means the objective is the
	task's reward rather than a latent distance, so planning numbers from here are not
	directly comparable with the paper's.

	### Assumptions this wrapper makes about the task

	* The task's observation is a dict/TensorDict. `cfg.visual_key` names the RGB frames
	  and `cfg.proprio_keys` the vector observations that make up the repo's `proprio`;
	  their concatenated width must match what the WM's proprio encoder was trained on.
	* The task's frame stack (`task.cfg.num_frames`) equals the WM's `num_hist`, so one
	  observation from the task is exactly one predictor window.
	* `cfg.sparse_imagination_cfg.frameskip > 1` means one WM step covers several env steps,
	  and one WM action is `frameskip` env actions concatenated. There are two ways a task can
	  meet that, and `_validate_task_compatibility` picks between them by action width:

	  - The task's own action *is* the concatenated chunk, because its env applies
	    `frame_skip` primitive actions per step (`custom_maniskill_tasks.FrameSkip`, as
	    `PushTTask` and `ManiSkillTask` both do) and its dataset concatenates the same
	    ones. One WM step is then exactly one env step, and the planner searches the space
	    the WM was trained on. This is what a checkpoint trained at `frameskip > 1` needs
	    in order to be run faithfully — set `task.cfg.frame_skip` to the same value.
	  - The task's action is a single primitive action, and `_to_sparse_action` tiles it
	    over the skip. A held-action approximation: it can only express a constant action
	    per WM step, and the WM's latent advances `frameskip` primitive steps while the env
	    advances one, so the two drift apart over an episode.

	### One process, one repo

	Sparse Imagination, TC-WM and DINO-WM all import their own modules rootlessly under the
	same top-level names, so a single process can host one of them, never two. Composing
	`frozen_sparse_imagination_*` with `frozen_tc_wm_*` or `frozen_dino_wm_*` in one config
	raises from `s2p.lib.sparse_imagination_path` rather than quietly building one model out
	of another's classes.
	"""

	def __init__(self, cfg, task:OnlineTaskBase):
		super().__init__(cfg=cfg)
		self.cfg = cfg

		# This model's config node carries `_recursive_: false` — the only way to stop Hydra
		# instantiating the `_target_`s inside `cfg.sparse_imagination_cfg`, which are the
		# repo's own and are Trainer's to build. That flag applies to every kwarg of the node,
		# so `task` arrives as a config here rather than as a built task. Rebuild it: task
		# factories memoize on the config hash, so this is the same instance every other model
		# got, not a second env.
		self.task = instantiate(task) if not isinstance(task, TaskBase) else task
		self.sparse_config = self.cfg.sparse_imagination_cfg

		with _wandb_suppressed():
			Trainer, VWorldModel, VWorldModelDrop, default_transform = _import_sparse_imagination()
			trainer = Trainer(self.sparse_config)

		if trainer.predictor is None:
			raise ValueError(
				"This world model was built with has_predictor=false, so it has no dynamics. "
				"It cannot be used as a world model."
			)

		self.drop = self.cfg.drop
		self.sparse_wm = self._build_world_model(trainer, VWorldModelDrop)
		self.sparse_wm = self.sparse_wm.to(self.cfg.device)

		self.num_hist = self.sparse_wm.num_hist
		self.concat_dim = self.sparse_wm.concat_dim
		self.frameskip = self.sparse_config.frameskip
		self._derive_latent_geometry()

		# How the task's observation dict maps onto the repo's (visual, proprio) pair.
		self.visual_key = self.cfg.visual_key
		self.dino_feature_key = self.cfg.dino_feature_key
		self.proprio_keys = list(self.cfg.proprio_keys)

		# Live env frames arrive as raw uint8 CHW; the dataset put its frames through
		# `default_transform` (resize, centre crop, Normalize(0.5, 0.5)) after scaling to
		# [0, 1]. DinoV3Encoder.forward assumes that [-1, 1] convention, so reproduce it.
		self.image_transform = default_transform(self.sparse_config.img_size)

		# Action/proprio normalization stats. The repo normalizes both against dataset
		# statistics before they ever reach the model, so anything coming from the task
		# (raw env actions, raw proprio) has to be put through the same map or the
		# predictor sees out-of-distribution inputs.
		self._register_normalization_stats(trainer)
		self._validate_task_compatibility()

		# The private stream `resample_kept_patches` draws subsets from. Not a tensor this
		# model owns as state, so not a buffer.
		self._patch_generator = None
		if self.drop and self.cfg.kept_patch_seed is not None:
			self._patch_generator = torch.Generator()
			self._patch_generator.manual_seed(int(self.cfg.kept_patch_seed))
			# Redraw the subset VWorldModelDrop.__init__ took from the ambient RNG, so that
			# the one this model actually plans with is a function of the seed alone.
			with _global_rng_from(self._patch_generator):
				self.sparse_wm.reset_random_patches()

	# ------------------------------------------------------------------ setup helpers

	def _build_world_model(self, trainer, VWorldModelDrop) -> torch.nn.Module:
		"""
		The world model to plan with: Trainer's own, or the sparse one built over its weights.

		Without `cfg.drop` this is just `trainer.model`, unwrapped. With it, the trained
		submodules are rehoused in a `VWorldModelDrop` exactly as `plan.py::load_model` does
		— including dropping the decoder, since a latent holding `plan_num_kept_patches` of
		the patch grid is not something the VQVAE can lay back out as an image (see `decode`).
		Trainer has no route to that class: `init_models` always instantiates
		`cfg.model._target_`, and the repo string-substitutes `VWorldModel` -> `VWorldModelDrop`
		at planning time rather than configuring it.

		`VWorldModelDrop.__init__` mutates the predictor it is handed (`num_patches` becomes
		the kept count, and the causal mask is rebuilt to match), which is why Trainer's own
		model is discarded here rather than kept alongside: the two cannot share a predictor
		and both be correct.

		The proprio/action embedding widths come off the built modules rather than from the
		config — the same choice `Trainer.init_models` makes, and the safe one, since they are
		what `separate_emb` slices by and a checkpoint's encoder need not agree with the
		config it is being rebuilt under.
		"""
		if not self.drop:
			if self.cfg.plan_num_kept_patches is not None:
				raise ValueError(
					"cfg.plan_num_kept_patches is only meaningful with cfg.drop: true, which "
					"selects the sparse world model. Set drop: true, or leave the kept-patch "
					"count null to plan on the full token grid."
				)
			model = _unwrap(trainer.model)
			for name in _SPARSE_MODULE_KEYS:
				submodule = getattr(model, name, None)
				if submodule is not None:
					setattr(model, name, _unwrap(submodule))
			return model

		if self.cfg.plan_num_kept_patches is None:
			raise ValueError(
				"cfg.drop: true plans on a random subset of the patch tokens, so it needs "
				"cfg.plan_num_kept_patches. conf/plan.yaml's sparse default is 98 of 196."
			)
		if self.sparse_config.concat_dim != 1:
			raise ValueError(
				f"cfg.drop: true is only supported with concat_dim=1, not "
				f"{self.sparse_config.concat_dim}. With concat_dim=0 the proprio and action "
				"are two extra tokens, but VWorldModelDrop sets the predictor's num_patches "
				"to the kept *patch* count alone -- so the frame-wise positional embedding "
				"ViTPredictorWithoutPE expands would be two tokens short of the sequence."
			)

		# `plan.py::load_model` gets here by string-substituting the `_target_` and handing it
		# back to Hydra; the class is already in hand, so it is called directly instead. That
		# keeps the repo's own guards (a sparse model needs a frame-only-PE predictor)
		# readable rather than wrapped in an InstantiationException, and it does not depend on
		# the `_target_` string being spelled a particular way -- only on its naming the class
		# whose kwargs these are, which is checked instead.
		model_target = str(self.sparse_config.model._target_)
		if not model_target.endswith("VWorldModel"):
			raise ValueError(
				f"cfg.drop: true rehouses the trained submodules in VWorldModelDrop, which "
				f"takes VWorldModel's arguments -- but this config builds {model_target!r}. "
				"Plan with drop: false, or teach this wrapper about the variant."
			)
		model_kwargs = {
			key: value for key, value in self.sparse_config.model.items() if key != "_target_"
		}
		return VWorldModelDrop(
			**model_kwargs,
			encoder=_unwrap(trainer.encoder),
			proprio_encoder=_unwrap(trainer.proprio_encoder),
			action_encoder=_unwrap(trainer.action_encoder),
			predictor=_unwrap(trainer.predictor),
			decoder=None,
			proprio_dim=_unwrap(trainer.proprio_encoder).emb_dim,
			action_dim=_unwrap(trainer.action_encoder).emb_dim,
			concat_dim=self.sparse_config.concat_dim,
			num_action_repeat=self.sparse_config.num_action_repeat,
			num_proprio_repeat=self.sparse_config.num_proprio_repeat,
			plan_num_kept_patches=int(self.cfg.plan_num_kept_patches),
		)

	def _derive_latent_geometry(self) -> None:
		"""
		Read the token grid off the predictor, and the token width off its position embedding.

		The count has to come from `predictor.num_patches` rather than from the shape of
		`pos_embedding`, which is what `DINOWorldModel` divides down: the sparse baselines run
		`ViTPredictorWithoutPE`, whose embedding is `(1, num_frames, dim)` — one vector per
		frame, expanded across patches at forward time by exactly this attribute. It is also
		the authority for the dense case, since `train.py` passes the same number in when it
		builds the ViT, and `VWorldModelDrop.__init__` overwrites it with the kept count.

		The width is unambiguous either way: `pos_embedding`'s last dimension is the
		predictor's, and it is cross-checked against what the model believes a token is made
		of, which is what every slice in this class and in `separate_emb` depends on.
		"""
		predictor = self.sparse_wm.predictor
		self.tokens_per_frame = int(predictor.num_patches)
		self.token_dim = predictor.pos_embedding.shape[-1]
		self.latent_dim = self.num_hist * self.tokens_per_frame * self.token_dim

		if self.token_dim != self.sparse_wm.emb_dim:
			raise ValueError(
				f"The predictor takes {self.token_dim}-wide tokens, but this world model "
				f"assembles {self.sparse_wm.emb_dim}-wide ones (encoder "
				f"{self.sparse_wm.encoder.emb_dim} + proprio {self.sparse_wm.proprio_dim} + "
				f"action {self.sparse_wm.action_dim}, at concat_dim={self.concat_dim}). The "
				"checkpoint and the config it was rebuilt from describe different models."
			)

		# A dense predictor bakes the full sequence length into its position embedding, so it
		# can say whether `num_patches` is the number this checkpoint was trained with. A
		# frame-only one cannot -- there is nothing in it that scales with the patch count,
		# which is the property the sparse baselines rely on.
		full_pos_embedding = predictor.pos_embedding.shape[1] != predictor.num_frames
		if full_pos_embedding and predictor.pos_embedding.shape[1] != self.num_hist * self.tokens_per_frame:
			raise ValueError(
				f"The predictor's position embedding covers {predictor.pos_embedding.shape[1]} "
				f"tokens, but its num_patches ({self.tokens_per_frame}) over {self.num_hist} "
				f"frames is {self.num_hist * self.tokens_per_frame}."
			)

		# Width of one frame's action slot, in the same units `separate_emb` returns:
		# a whole extra token for concat_dim=0, the un-tiled action embedding otherwise.
		if self.concat_dim == 0:
			self.action_slot_dim = self.token_dim
		else:
			self.action_slot_dim = self.sparse_wm.action_dim // self.sparse_wm.num_action_repeat

	def _register_normalization_stats(self, trainer) -> None:
		# TrajSubset forwards attribute lookups to the underlying PushBlockDataset, which
		# is where the stats live (they are ones/zeros when normalize_action is false).
		dataset = trainer.train_traj_dset
		expected_action_dim = self.sparse_wm.action_encoder.in_chans
		primitive_action_dim = max(1, expected_action_dim // self.frameskip)
		proprio_dim = self.sparse_wm.proprio_encoder.in_chans

		def stat(name, default):
			value = getattr(dataset, name, None)
			return default if value is None else torch.as_tensor(value, dtype=torch.float32)

		# The action statistics on one of these trajectory datasets are per *primitive*
		# action: it normalizes each one and only then concatenates `frameskip` of them
		# (TrajSlicerDataset). Tiling them here to the width of a world model action makes
		# that the same as normalizing the concatenated action in one go, which is what
		# `_to_sparse_action` then does — whichever way the action reached full width.
		action_mean = stat("action_mean", torch.zeros(primitive_action_dim))
		action_std = stat("action_std", torch.ones(primitive_action_dim))
		if action_mean.shape[-1] * self.frameskip == expected_action_dim:
			action_mean = action_mean.repeat(self.frameskip)
			action_std = action_std.repeat(self.frameskip)

		self.register_buffer("action_mean", action_mean.to(self.cfg.device))
		self.register_buffer("action_std", action_std.to(self.cfg.device))
		self.register_buffer("proprio_mean", stat("proprio_mean", torch.zeros(proprio_dim)).to(self.cfg.device))
		self.register_buffer("proprio_std", stat("proprio_std", torch.ones(proprio_dim)).to(self.cfg.device))

	def _validate_task_compatibility(self) -> None:
		"""
		Fail loudly at construction on the mismatches that would otherwise surface as an
		unreadable shape error deep inside the predictor, or — worse — not at all.
		"""
		observation_dimension = self.task.observation_dimension
		if not isinstance(observation_dimension, dict):
			raise ValueError(
				"This world model needs both an image and a proprioceptive vector, so the "
				f"task must expose a dict observation. Got "
				f"{type(observation_dimension).__name__}."
			)

		# One task observation must be exactly one predictor window.
		# Read the stack depth off whichever visual leaf the task carries: a recording that
		# ships precomputed DINO features need not also carry the frames they came from. Both
		# live under `obs` and are frame-stacked alike, so they agree wherever both exist.
		stack_key = self.visual_key if self.visual_key in observation_dimension else self.dino_feature_key
		if stack_key not in observation_dimension:
			raise ValueError(
				f"{type(self).__name__} needs a frame-stacked visual leaf to size its predictor "
				f"window, but the task exposes neither {self.visual_key!r} nor "
				f"{self.dino_feature_key!r}. Got {sorted(observation_dimension)}."
			)
		frame_stack = observation_dimension[stack_key][0]
		if frame_stack != self.num_hist:
			raise ValueError(
				f"Task stacks {frame_stack} frames but the world model's num_hist is "
				f"{self.num_hist}. Set task.cfg.num_frames = {self.num_hist}."
			)

		missing = [key for key in self.proprio_keys if key not in observation_dimension]
		if missing:
			raise ValueError(
				f"cfg.proprio_keys {missing} are not in the task's observation "
				f"({sorted(observation_dimension)})."
			)
		proprio_dim = sum(observation_dimension[key][-1] for key in self.proprio_keys)
		expected_proprio_dim = self.sparse_wm.proprio_encoder.in_chans
		if proprio_dim != expected_proprio_dim:
			raise ValueError(
				f"cfg.proprio_keys {self.proprio_keys} give a {proprio_dim}-dim proprio "
				f"vector, but this world model's proprio encoder was trained on "
				f"{expected_proprio_dim} dims. The PushCube dataset concatenates "
				"obs/agent/qpos, obs/agent/qvel and obs/extra/tcp_pose (9+9+7=25); either "
				"expose the same fields through the task's dataset_structure, or retrain "
				"the world model on the fields the task does expose."
			)

		if self.drop and self.tokens_per_frame > self.sparse_wm.patch_num:
			raise ValueError(
				f"cfg.plan_num_kept_patches is {self.tokens_per_frame}, but this encoder "
				f"produces only {self.sparse_wm.patch_num} patch tokens per frame."
			)

		# Either the task already acts in chunks of `frameskip` primitive actions, in which
		# case its action goes to the world model as it is, or it acts one primitive action
		# at a time and `_to_sparse_action` tiles it. See the class docstring for why the
		# first is the one a checkpoint trained at frameskip > 1 needs. At frameskip 1 the
		# two are the same thing.
		expected_action_dim = self.sparse_wm.action_encoder.in_chans
		task_action_dim = self.task.action_dimension[-1]
		self.action_is_chunked = task_action_dim == expected_action_dim
		if not self.action_is_chunked and task_action_dim * self.frameskip != expected_action_dim:
			raise ValueError(
				f"This world model's action encoder was trained on {expected_action_dim} dims, "
				f"which frameskip {self.frameskip} splits into {self.frameskip} x "
				f"{expected_action_dim // self.frameskip}. The task's action is "
				f"{task_action_dim}-dim, which is neither. Set task.cfg.frame_skip = "
				f"{self.frameskip} so the env acts in chunks of the right width, or check that "
				"the task and the checkpoint describe the same env."
			)

	# ------------------------------------------------------- latent (un)flattening

	def _flatten_latent(self, z:torch.Tensor, batch_dims:torch.Size) -> torch.Tensor:
		"""(b, num_hist, tokens, dim) -> (*batch_dims, latent_dim)"""
		return z.reshape(*batch_dims, self.latent_dim)

	def _unflatten_latent(self, s:torch.Tensor) -> torch.Tensor:
		"""(*batch_dims, latent_dim) -> (prod(batch_dims), num_hist, tokens, dim)"""
		if s.shape[-1] != self.latent_dim:
			raise ValueError(
				f"Expected a latent of width {self.latent_dim}, got {s.shape[-1]}."
			)
		return s.reshape(-1, self.num_hist, self.tokens_per_frame, self.token_dim)

	# ------------------------------------------------------------ observation adaption

	def _prepare_visual(self, visual:torch.Tensor) -> torch.Tensor:
		"""(b, t, 3, H, W) raw frames -> the [-1, 1] convention the encoder expects."""
		b, t = visual.shape[:2]
		x = rearrange(visual, "b t c h w -> (b t) c h w")
		if not torch.is_floating_point(x):
			x = x.float() / 255.0
		x = self.image_transform(x)
		return rearrange(x, "(b t) c h w -> b t c h w", b=b)

	def _to_sparse_obs(self, observation) -> typing.Tuple[dict, torch.Size]:
		"""
		Adapt one of this package's observations to the repo's obs dict.

		Observations here are `(*batch_dims, S, *feature_dims)` with the frame stack `S`
		innermost and `batch_dims` typically `(T, B)`; the repo wants `(b, t, ...)` with a
		single leading batch dim. Since `S == num_hist`, the mapping is just "collapse the
		batch dims and let the frame stack be the repo's time axis".
		"""
		try:
			keys = set(observation.keys())
		except AttributeError:
			raise ValueError(
				"This world model needs both an image and a proprioceptive vector, so "
				f"`encode` must be given a dict observation. Got "
				f"{type(observation).__name__}."
			)

		sparse_obs = {}
		# The dataset ships DINOv3 patch features precomputed with the very backbone this
		# model's encoder holds, and `encode_visual` short-circuits to them when present —
		# that is the whole reason offline training never runs the encoder. They are absent
		# from live env observations, which fall through to the visual path below. Under
		# `drop` the kept-patch subset is taken *after* this, so the two paths select the
		# same tokens.
		if self.dino_feature_key is not None and self.dino_feature_key in keys:
			features = observation[self.dino_feature_key]      # (*batch, S, P, D)
			batch_dims = features.shape[:-3]
			sparse_obs["dino_patch_features"] = features.reshape(-1, *features.shape[-3:]).float()
		else:
			visual = observation[self.visual_key]              # (*batch, S, 3, H, W)
			batch_dims = visual.shape[:-4]
			sparse_obs["visual"] = self._prepare_visual(visual.reshape(-1, *visual.shape[-4:]))

		proprio = torch.cat([observation[key] for key in self.proprio_keys], dim=-1)
		proprio = proprio.reshape(-1, *proprio.shape[-2:]).float()
		sparse_obs["proprio"] = (proprio - self.proprio_mean) / self.proprio_std

		return sparse_obs, batch_dims

	def _to_sparse_action(self, action:torch.Tensor) -> torch.Tensor:
		"""
		(*batch_dims, action_dim) task actions -> (prod(batch_dims), 1, world model action dim)
		normalized world model actions.

		A world model action is `frameskip` consecutive primitive actions concatenated (see
		TrajSlicerDataset). A task whose env already acts in those chunks supplies one
		directly; a task that acts one primitive action at a time has it tiled to fill the
		skip. Either way normalization is applied at full width, against the tiled statistics
		`_register_normalization_stats` built — which for the tiling path is the same result
		as the dataset's own "normalize each action, then concatenate".
		"""
		a = action.float().reshape(-1, 1, action.shape[-1])
		if not self.action_is_chunked and self.frameskip > 1:
			a = a.repeat(1, 1, self.frameskip)
		return (a - self.action_mean) / self.action_std

	# ---------------------------------------------------------------- latent assembly

	def _assemble_latent(self, z_obs:dict, action_embedding:torch.Tensor) -> torch.Tensor:
		"""
		Mirror of `VWorldModel.encode`, but taking an already-embedded action.

		`encode` embeds actions itself, which is no use here: the action that belongs in
		frame `i`'s slot is usually not one we hold in raw form (it comes out of the
		previous latent), and the newest one has to be tiled the same way regardless.
		"""
		if self.concat_dim == 0:
			return torch.cat(
				[z_obs["visual"], z_obs["proprio"].unsqueeze(2), action_embedding.unsqueeze(2)],
				dim=2,
			)

		num_patches = z_obs["visual"].shape[2]
		proprio_tiled = repeat(z_obs["proprio"].unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.sparse_wm.num_proprio_repeat)
		action_tiled = repeat(action_embedding.unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		action_repeated = action_tiled.repeat(1, 1, 1, self.sparse_wm.num_action_repeat)
		return torch.cat([z_obs["visual"], proprio_repeated, action_repeated], dim=3)

	def _zero_action_embedding(self, batch_size:int, num_frames:int, device) -> torch.Tensor:
		"""
		`encode_act` of the zero action — what an unknown or not-yet-chosen slot holds.

		Writing literal zeros there instead would be out of distribution: `encode_act` is a
		1x1 convolution *with a bias*, so no action embeds to the zero vector and the
		predictor has never seen one. The difference is the bias, which is small in
		magnitude but sits in every one of the tokens' action channels, and at the start of
		an episode it is two of the three frames the predictor attends over.

		The zero action is normalized the same way a real one is (`_to_sparse_action`), so
		this is genuinely "the agent took the null action", not "these channels are blank".
		For a delta action space that reads as "nothing was commanded to move", which is the
		honest stand-in for a step that never happened.

		Returns `(batch_size, num_frames, action_slot_dim)`, matching `_action_slots`.
		"""
		zero = torch.zeros(
			batch_size, num_frames, self.sparse_wm.action_encoder.in_chans, device=device
		)
		return self.sparse_wm.encode_act((zero - self.action_mean) / self.action_std)

	def _clear_action_slot(self, z:torch.Tensor) -> torch.Tensor:
		"""
		Reset the action slot of every frame of `z` to this class's "unset" marker.

		The marker is `encode_act(0)` rather than a literal zero, for the reason
		`_zero_action_embedding` gives, and it is written through the repo's own
		`replace_actions_from_z` so that both `concat_dim` layouts are handled by the one
		implementation that already knows them. Returns a copy, since the callers are
		holding latents the planners still own.

		In practice nothing reads a slot left in this state — `dynamics` overwrites the
		newest frame before predicting, and `_action_slots` drops it — but keeping it equal
		to what `_action_slots` writes is what makes a rolled-out latent and an encoded one
		interchangeable slot-for-slot.
		"""
		batch_size, num_frames = z.shape[:2]
		zero = torch.zeros(
			batch_size, num_frames, self.sparse_wm.action_encoder.in_chans, device=z.device
		)
		return self.sparse_wm.replace_actions_from_z(
			z.clone(), (zero - self.action_mean) / self.action_std
		)

	def _action_slots(self, previous_state, action, batch_size:int, device) -> torch.Tensor:
		"""
		Build the per-frame action embeddings for a freshly encoded window.

		Frame `i` of a window carries the action applied *at* frame `i`, taking it to frame
		`i+1`. That is the alignment the model was trained on — the recording puts
		`states[j]` before `actions[j]` — and the one `VWorldModel.rollout` maintains, where
		each newly predicted frame receives the next action of the plan before being
		predicted from.

		So the newest frame's slot is not something an observation can supply: it is the
		action nobody has chosen yet, the planner's decision variable, and `dynamics` fills
		it in for the duration of one prediction. This leaves `num_hist - 1` past actions to
		place, and `action` — the action that carried the episode *into* the newest frame —
		belongs to the second-newest slot, not the newest. The ones older than that are
		lifted out of `previous_state`, whose window is the same frames shifted back by one:
		`[t-2, t-1, t]` here, `[t-3, t-2, t-1]` there, so dropping its oldest frame and its
		own unset newest slot leaves exactly the ones needed.

		Everything unknown holds `encode_act(0)` — see `_zero_action_embedding` for why that
		rather than a literal zero. Two things are unknown: the unset newest slot, and the
		frames `FrameStack` padded at the start of an episode, whose actions simply do not
		exist yet. The first washes out because `dynamics` overwrites it, the second after
		`num_hist` steps. The convention propagates on its own: a window whose padded slot
		holds `encode_act(0)` hands that same value forward through `previous_actions`.

		Note that the action slots survive a change of kept-patch subset untouched: they are
		one embedding tiled across whichever tokens a frame has, so `separate_emb` reads them
		back off token 0 whatever was kept. A `previous_state` from before a resample is
		still a usable source of action history, even though its visual tokens are not
		comparable with the current ones.
		"""
		# Nothing is known: an offline encoding that was handed no action at all, or a
		# window whose only slot is the unset one (`num_hist == 1`, where `action` was
		# applied at a frame that is no longer in the window).
		if action is None or self.num_hist == 1:
			return self._zero_action_embedding(batch_size, self.num_hist, device)

		unset = self._zero_action_embedding(batch_size, 1, device)
		applied = self.sparse_wm.encode_act(self._to_sparse_action(action))   # (b, 1, E)
		if previous_state is None:
			# `num_hist == 2` leaves nothing older than `applied`; the convolution would
			# handle a zero-length sequence, but an explicit empty tensor is clearer.
			older = (
				self._zero_action_embedding(batch_size, self.num_hist - 2, device)
				if self.num_hist > 2
				else torch.zeros(batch_size, 0, self.action_slot_dim, device=device)
			)
		else:
			_, previous_actions = self.sparse_wm.separate_emb(self._unflatten_latent(previous_state))
			older = previous_actions[:, 1:-1]
		return torch.cat([older, applied, unset], dim=1)

	# ------------------------------------------------------------------ encoder model

	def encode(self, observation, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
		sparse_obs, batch_dims = self._to_sparse_obs(observation)
		z_obs = self.sparse_wm.encode_obs(sparse_obs)

		batch_size, num_frames = z_obs["proprio"].shape[:2]
		if num_frames != self.num_hist:
			raise ValueError(
				f"Observation carries {num_frames} stacked frames, but the world model's "
				f"predictor window is {self.num_hist} frames wide."
			)

		action_embedding = self._action_slots(
			previous_state, action, batch_size, z_obs["proprio"].device
		)
		z = self._assemble_latent(z_obs, action_embedding)
		return self._flatten_latent(z, batch_dims)

	def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		"""
		The encoding path, for FLOP counting.

		Sparse or not, this is the same computation: the backbone runs on the whole frame and
		the kept-patch subset is an index into its output. The paper's saving is in the
		predictor (`get_dynamics_function`), which is where the token count actually shows up
		— so a `drop98` model and a dense one should and do report the same encoder cost.

		Note this measures the *live* path. Offline batches carry precomputed
		`dino_patch_features` and skip the backbone entirely, so their true encoding cost is
		only the proprio and action embeddings.
		"""
		encoder_module = _SparseEncoderForFLOPS(
			encoder=self.sparse_wm.encoder,
			proprio_encoder=self.sparse_wm.proprio_encoder,
			action_encoder=self.sparse_wm.action_encoder,
			encoder_transform=self.sparse_wm.encoder_transform,
			keep_patches_idx=getattr(self.sparse_wm, "keep_patches_idx", None),
		)
		return (
			(self.num_hist, 3, self.sparse_config.img_size, self.sparse_config.img_size),
			encoder_module,
		)

	# ----------------------------------------------------------------- dynamics model

	def write_action_into_newest_frame(self, z:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
		"""
		`z` with `action` written into its newest frame's action slot.

		The first of the three steps `dynamics` runs, factored out because a reward head over
		this latent has to condition on the action the same way the predictor does -- see
		`s2p.models.dino_wm_reward_model.DINOWMRewardModel.reward`, which is written against
		this method rather than against any one wrapper's inner model so that the same head
		serves every DINO-WM-family wrapper. `sparse_wm` and `_to_sparse_action` are what
		differ between them.

		Takes an unflattened window `(batch, num_frames, tokens, dim)` and returns one, leaving
		the caller's `z` untouched: Sparse Imagination's `replace_actions_from_z` writes in place, so the
		frame it is handed is cloned first.
		"""
		newest = self.sparse_wm.replace_actions_from_z(
			z[:, -1:].clone(), self._to_sparse_action(action)
		)
		return torch.cat([z[:, :-1], newest], dim=1)

	def dynamics(self, s:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
		batch_dims = s.shape[:-1]
		z = self._unflatten_latent(s).clone()

		# Write the action into the newest frame's slot, predict, and slide the window by
		# one — the same three steps as one iteration of `VWorldModel.rollout`. The clone
		# above matters: `replace_actions_from_z` writes in place, and the planners reuse
		# the latent they pass in (MPPI scores the reward head off it after this returns).
		#
		# `rollout` also calls `_restore_predictor_mask()` before it starts, which this does
		# not: the mask is only ever perturbed by `_apply_train_random_drop`, which runs
		# inside `forward` and restores it in a `finally`. Nothing at inference touches it,
		# and re-uploading a (num_hist * tokens)^2 mask on every step of every rollout would
		# be pure cost on the planner's hot path. It is set once, correctly, when the world
		# model is constructed.
		z = self.write_action_into_newest_frame(z, a)

		z_predicted = self.sparse_wm.predict(z)

		# The predictor emits a full token, action channels included, for the frame it just
		# predicted — but the action applied *at* that frame is the next one to be chosen,
		# so those channels are cleared rather than kept. The repo does the same by writing
		# the plan's next action over them (`rollout`), and the next `dynamics` call does
		# exactly that; clearing them in the meantime is what keeps a rolled-out latent
		# interchangeable with an encoded one, which MPPI relies on when it scores the
		# reward head off whichever of the two it is holding.
		z_new = self._clear_action_slot(z_predicted[:, -1:])
		z_next = torch.cat([z[:, 1:], z_new], dim=1)
		return self._flatten_latent(z_next, batch_dims)

	def get_dynamics_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		dynamics_module = _SparseDynamicsForFLOPS(
			predictor=self.sparse_wm.predictor,
			action_encoder=self.sparse_wm.action_encoder,
			latent_dim=self.latent_dim,
			num_hist=self.num_hist,
			tokens_per_frame=self.tokens_per_frame,
			token_dim=self.token_dim,
		)
		return (
			(self.latent_dim + self.sparse_wm.action_encoder.in_chans,),
			dynamics_module,
		)

	# --------------------------------------------------------------- sparse imagination

	def resample_kept_patches(self) -> None:
		"""
		Draw a fresh random subset of the patch tokens to imagine on.

		A no-op without `cfg.drop`, where there is no subset to draw.

		**Nothing in this package calls this.** The subset drawn at construction is the one
		the model plans with for its whole lifetime, and that is the intended behaviour, not
		an omission. It is exposed because the repo does resample, and because a controlled
		experiment might want to.

		Which patches are kept is part of what a latent *means* — two latents are comparable
		only if they were built from the same tokens — so a resample invalidates every latent
		taken before it, including ones this model does not know about. That is what makes an
		automatic hook unsafe here: the reward head (`DINOWMRewardModel`) is fitted against
		latents from one subset, and MPPI is handed a starting latent that was encoded before
		`plan` was entered, so the repo's per-replan `MPCPlanner._refresh_random_patches` has
		no equivalent that is not also a way to score a rollout against a head that was
		trained on different tokens.

		A caller that does invoke this must re-`encode` anything it is holding. Only the
		*visual* tokens go stale — the action slots are one embedding tiled across whatever
		tokens a frame has, so an old `previous_state` is still a usable action history (see
		`_action_slots`).

		With `cfg.kept_patch_seed` set, subsets come from a private generator rather than the
		global RNG, so the sequence is reproducible and does not shift when anything else in
		the process draws (see `_global_rng_from`).
		"""
		if not self.drop:
			return
		if self._patch_generator is None:
			self.sparse_wm.reset_random_patches()
		else:
			with _global_rng_from(self._patch_generator):
				self.sparse_wm.reset_random_patches()

	# ------------------------------------------------------------------------ extras

	def decode(self, s:torch.Tensor) -> torch.Tensor:
		"""
		Render a latent window back to images, for inspecting rollouts. Returns
		`(*batch_dims, num_hist, 3, H, W)`; requires the model to have a decoder.

		Never available under `cfg.drop`. The VQVAE decoder lays its input tokens out as a
		square patch grid, and a sparse latent holds an arbitrary subset of that grid with no
		record of where the tokens came from — `keep_patches_idx` says which they were, but
		the dropped ones have no values to put back. `plan.py::load_model` reaches the same
		conclusion by force (`has_decoder = False` whenever `drop`), which is why
		`_build_world_model` does not carry a decoder over.
		"""
		if self.sparse_wm.decoder is None:
			raise ValueError(
				"This world model has no decoder — either it was built with "
				"has_decoder=false, or cfg.drop dropped it because a sparse token subset "
				"cannot be laid out as an image."
			)
		batch_dims = s.shape[:-1]
		obs, _ = self.sparse_wm.decode(self._unflatten_latent(s))
		visual = obs["visual"]
		return visual.reshape(*batch_dims, *visual.shape[1:])

	def train(self, mode:bool=True):
		"""
		Keep a frozen world model in eval mode whatever mode the agent is put in.

		The ViT predictor carries dropout (0.1 in conf/predictor/vit_nope.yaml), so a world
		model left in train mode returns a *different* rollout for the same state and
		action on every call — noise injected straight into every planner score, and a
		different function from the one every evaluator sees under `model.eval()`.
		"""
		super().train(mode)
		if self.cfg.freeze:
			self.sparse_wm.eval()
		return self

	def requires_grad_(self, requires_grad):
		return super().requires_grad_(requires_grad and not self.cfg.freeze)

	def save_to_file(self, filepath) -> None:
		# Frozen: these weights are still exactly the file the vendored Trainer resumed
		# from, so point at it instead of copying it. See s2p.lib.checkpointing.
		source = resume_folder_checkpoint(self.sparse_config.resume_folder) if self.cfg.freeze else None
		if source is not None:
			save_checkpoint_reference(filepath, source, type(self).__name__)
			return

		torch.save(self.state_dict(), filepath)

	def load_from_file(self, filepath:str) -> None:
		# May be a reference written by `save_to_file` above rather than weights.
		filepath = resolve_checkpoint_path(filepath)
		checkpoint = torch.load(filepath, map_location=self.cfg.device, weights_only=False)

		# A native checkpoint is a dict of whole pickled modules keyed by role
		# (Trainer.save_ckpt / plan.load_ckpt), not a state dict — swap them in wholesale.
		# Anything else is assumed to be one of ours, written by `save_to_file`.
		is_native = isinstance(checkpoint, dict) and any(
			isinstance(checkpoint.get(key), torch.nn.Module) for key in _SPARSE_MODULE_KEYS
		)
		if not is_native:
			self.load_state_dict(checkpoint)
			return

		for key in _SPARSE_MODULE_KEYS:
			module = checkpoint.get(key)
			if isinstance(module, torch.nn.Module):
				if key == "decoder" and self.drop:
					# See `decode`: the sparse model deliberately has none, and adopting one
					# here would leave `decode` raising a shape error instead of the
					# explanation it currently gives.
					continue
				setattr(self.sparse_wm, key, _unwrap(module).to(self.cfg.device))

		# A swapped-in predictor arrives with the token count and mask it was *trained* with,
		# which for a sparse model is not the one it plans with. Re-apply both, then re-derive
		# the geometry every reshape in this class is keyed off.
		if self.drop:
			self.sparse_wm.predictor.num_patches = self.sparse_wm.num_kept_patches
			self.sparse_wm._reset_predictor_mask(num_patches=self.sparse_wm.num_kept_patches)
		else:
			self.sparse_wm._reset_predictor_mask()
		self._derive_latent_geometry()


class _SparseEncoderForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats the encoding of one observation window. Used for
	counting the FLOPs of the encoding path only.

	The kept-patch selection is included so the reported output shape is the one the
	predictor actually sees, but it is an index operation and contributes nothing: the
	backbone above it has already run on every patch. That is the honest picture — sparse
	imagination makes the *predictor* cheaper, not the encoder.
	"""
	def __init__(self, encoder, proprio_encoder, action_encoder, encoder_transform, keep_patches_idx):
		super().__init__()
		self.encoder = encoder
		self.proprio_encoder = proprio_encoder
		self.action_encoder = action_encoder
		self.encoder_transform = encoder_transform
		self.register_buffer("keep_patches_idx", keep_patches_idx, persistent=False)

	def forward(self, visual:torch.Tensor):
		b, t = visual.shape[:2]
		x = rearrange(visual, "b t c h w -> (b t) c h w")
		visual_embedding = self.encoder(self.encoder_transform(x))
		visual_embedding = rearrange(visual_embedding, "(b t) p d -> b t p d", b=b)
		if self.keep_patches_idx is not None:
			visual_embedding = visual_embedding[:, :, self.keep_patches_idx, :]

		# Values do not affect FLOPs; only the shapes the two 1-D convolutions see do.
		device = visual.device
		self.proprio_encoder(torch.zeros(b, t, self.proprio_encoder.in_chans, device=device))
		self.action_encoder(torch.zeros(b, t, self.action_encoder.in_chans, device=device))
		return visual_embedding


class _SparseDynamicsForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats one prediction step from a flat (latent, action) input.
	Used for counting the FLOPs of the dynamics only — and the place where a sparse model is
	cheaper than a dense one, since the predictor's attention is quadratic in the token count
	this latent carries.
	"""
	def __init__(self, predictor, action_encoder, latent_dim, num_hist, tokens_per_frame, token_dim):
		super().__init__()
		self.predictor = predictor
		self.action_encoder = action_encoder
		self.latent_dim = latent_dim
		self.num_hist = num_hist
		self.tokens_per_frame = tokens_per_frame
		self.token_dim = token_dim
		self.input_size = (latent_dim + action_encoder.in_chans,)

	def forward(self, latent_action:torch.Tensor):
		latent, action = latent_action.split(
			(self.latent_dim, self.action_encoder.in_chans), dim=-1
		)
		# Embedding the action and writing it into the newest frame's slot; the write
		# itself is a copy and contributes nothing, so only the embedding is replayed.
		self.action_encoder(action.unsqueeze(1))
		z = latent.reshape(-1, self.num_hist * self.tokens_per_frame, self.token_dim)
		return self.predictor(z)


_registry: dict = {}

def get_or_create(cfg, task) -> "SparseImaginationWorldModel":
	key = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = SparseImaginationWorldModel(cfg=cfg, task=task)
	return _registry[key]
