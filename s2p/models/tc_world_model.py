import typing
import hashlib
import logging
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
from s2p.lib.tc_wm_path import ensure_tc_wm_importable
from s2p.lib.checkpointing import (
	resolve_checkpoint_path,
	resolve_resume_checkpoint,
	resume_folder_checkpoint,
	save_checkpoint_reference,
)


# Submodules that a native TC-WM checkpoint (Trainer.save_ckpt) stores as whole pickled
# nn.Modules rather than as state dicts, mirroring plan.py's ALL_MODEL_KEYS. Two more than
# DINO-WM's list: TC-WM adds the projector that makes its latent compact
# (`post_concat_projection`) and the linear map back into the backbone's embedding space
# (`emb_decoder`), plus the loss modules (`task_objectives`, an nn.ModuleList) which are
# training-only but travel in the same file.
_TCWM_MODULE_KEYS = (
	"encoder",
	"predictor",
	"decoder",
	"emb_decoder",
	"proprio_encoder",
	"action_encoder",
	"post_concat_projection",
	"task_objectives",
)


def _import_tc_wm():
	"""
	Import TC-WM's `Trainer` and image transform, putting its root on `sys.path` first.

	Deliberately *not* done at module import, unlike `s2p.models.dino_world_model`'s
	`from dino_wm.train import Trainer`, for two reasons:

	* TC-WM and DINO-WM claim the same rootless top-level module names (`models`,
	  `datasets`, `utils`, ...) and cannot both be importable in one process. Deferring
	  keeps merely *importing* this module harmless — see `s2p.lib.tc_wm_path`, which
	  raises with an explanation when the conflict is real.
	* TC-WM's `train.py` calls `wandb.login()` at module scope, i.e. as an import side
	  effect. Importing inside `_wandb_suppressed` is what turns that into a no-op
	  (`WANDB_MODE=disabled` makes `login` return immediately, before it reads any
	  credentials) rather than something that runs once, unsuppressed, whenever this file
	  is loaded.
	"""
	ensure_tc_wm_importable()
	from train import Trainer
	from datasets.img_transforms import default_transform
	return Trainer, default_transform


@contextlib.contextmanager
def _wandb_suppressed():
	"""
	Keep TC-WM's Trainer from claiming this process's wandb run.

	`Trainer.__init__` calls `wandb.init(project="minimal_wm")` unconditionally and then
	renames the run after the world model. wandb's `reinit` default is "return the previous
	run", so every later init — including the one EvaluationRunner makes into the S2P
	project — hands back that first run, and every S2P metric is logged into `minimal_wm`
	under the world model's name instead.

	Here Trainer is only a model factory, so its logging is unwanted outright rather than
	misdirected: `WANDB_MODE=disabled` turns its init into a NoopRun that still satisfies
	everything it does with the handle (`.id`, renaming, `watch`). The mode is cached in
	wandb's process-global setup, though — leaving it there would silently no-op the
	runner's own init too — so `teardown()` clears it on the way out.

	`WANDB_API_KEY` is saved and restored for the same reason: TC-WM's Trainer does
	`os.environ.setdefault("WANDB_API_KEY", "")`, and an empty key left behind in a process
	that had none would break the runner's own (real) init later, where the key would
	otherwise have come from `~/.netrc`.

	A run that was already active on entry is left alone: `reinit` means Trainer would have
	adopted (and renamed) it whatever this did, and tearing it down would be worse.
	"""
	preexisting_run = wandb.run
	previous_mode = os.environ.get("WANDB_MODE")
	previous_api_key = os.environ.get("WANDB_API_KEY")
	os.environ["WANDB_MODE"] = "disabled"
	try:
		yield
	finally:
		if preexisting_run is None:
			wandb.finish()
		for name, previous in (("WANDB_MODE", previous_mode), ("WANDB_API_KEY", previous_api_key)):
			if previous is None:
				os.environ.pop(name, None)
			else:
				os.environ[name] = previous
		if preexisting_run is None:
			wandb.teardown()


@contextlib.contextmanager
def _root_log_handlers_preserved():
	"""
	Undo the root-logger handler TC-WM's Trainer installs.

	`Trainer.__init__` attaches a `logging.FileHandler(cwd/train.log)` to the *root* logger
	and never removes it, so once this factory has run, every log record any part of this
	process emits — S2P's runners, evaluators, third-party libraries — is also written into
	a `train.log` describing a training run that is not happening. Handlers added while the
	body runs are dropped again here; ones that were already installed are untouched.
	"""
	preexisting = list(logging.getLogger().handlers)
	try:
		yield
	finally:
		root = logging.getLogger()
		for handler in list(root.handlers):
			if handler not in preexisting:
				root.removeHandler(handler)
				handler.close()


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
	"""
	Peel `torch.compile` / DDP wrappers off a module.

	Trainer ends `init_models` with `torch.compile(self.model)` and runs every submodule
	through `accelerator.prepare`, so the object it hands back may be an OptimizedModule
	(or a DistributedDataParallel) around the real VWorldModel. Attribute access proxies
	through those wrappers, but `state_dict` keys do not — they gain a `_orig_mod.` /
	`module.` prefix, which would make every checkpoint written here unloadable by the
	original TC-WM code. Unwrap once, up front, so both stay in the same namespace.
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


class TCWorldModel(
	EncoderModelBase,
	DynamicsModelBase,
	# PolicyModelBase,       # TC-WM has no policy
	torch.nn.Module
):
	"""
	TC-WM wrapper, exposing a trained `VWorldModel` through the encoder and dynamics
	interfaces the planners and losses in this package expect.

	The counterpart of `s2p.models.dino_world_model.DINOWorldModel`, and structured the same
	way, because TC-WM is a fork of DINO-WM: same ViT predictor over a history window, same
	frameskip/action-chunk convention. What differs is *what the latent is*, and everything
	below follows from that.

	### What "state" means here

	TC-WM's thesis is that a frozen foundation embedding is a scaffold, not a state space.
	So where DINO-WM's per-frame token is the DINOv3 patch embedding with proprio and action
	channels appended, TC-WM's is

		token = [ projected | action | proprio ]
		        [ P(visual, proprio) | tiled action embedding | tiled raw proprio ]

	where `P` is `post_concat_projection` — a small MLP compressing (visual embedding,
	proprio embedding) into `projected_dim` channels, a designated `alignment_dim`-wide
	subspace of which was aligned with the recorded state by InfoNCE during training. The
	trailing block is the *raw* (dataset-normalized) proprio vector rather than an
	embedding, and the predictor is trained to predict it, so a proprio prediction can be
	read straight out of a rolled-out latent.

	Like DINO-WM, the predictor is a ViT over `num_hist * num_patches` tokens, so a one-step
	prediction needs the whole window, and the latent this wrapper hands out is the
	*flattened window*

		s = flatten(z[:, -num_hist:])        # (num_hist * tokens_per_frame * token_dim,)

	`dynamics` writes the incoming action into the newest frame's action slot, calls
	`predictor`, and slides the window by one, which is precisely what `VWorldModel.rollout`
	does.

	Frame `i`'s action slot holds the action applied *at* frame `i`, taking it to frame
	`i+1`, so the newest frame's action is one nothing has chosen yet. Every latent this
	class hands out — from `encode` and from `dynamics` alike — therefore carries
	`encode_act(0)` there, and only `dynamics` fills it in, for the duration of one
	prediction. See `_action_slots`.

	That makes `s` Markov, at the cost of being large — though markedly less so than
	DINO-WM's, which is the point of the projection: for dinov3_vits16 at 224px with
	`projected_dim=256`, `action_emb_dim=32`, a 25-dim proprio and `num_hist=3` it is
	3*196*313 ≈ 184k floats *per sample*, against DINO-WM's 238k at the same settings. Still
	~0.4 GB for an MPPI population of 512, so turn `planner.cfg.num_samples` down
	accordingly.

	Only `concat_dim: 1` is supported, which is what every TC-WM config sets. See
	`__init__` for why `concat_dim: 0` is not merely unimplemented here.

	### Where the reward comes from

	Nowhere in here. This is an encoder and a dynamics model, and nothing else: it implements
	neither `RewardModelBase` nor `StateValueModelBase`, so `AgentModel` cannot be handed it
	as `reward_model` or `value_model` and `MPPIPlanner` will never ask it to score anything.

	That is a deliberate narrowing of what TC-WM's own code does. Upstream it plans by driving
	the latent towards a goal latent — `planning.objectives.create_objective_fn(mode="last")`,
	a terminal cost measured in the projected latent — which makes the model goal-conditioned:
	undefined until a goal is set, and re-aimed at every reset because the goal belongs to the
	episode rather than to the model. Reproducing that here would mean carrying a goal latent,
	a `set_goal`, and a task that can name its episode's goal.

	Instead the reward is a *separate* model fitted on top of this one:
	`s2p.models.dino_wm_reward_model.DINOWMRewardModel`, a cross-attention head over this
	latent window, trained by `s2p.losses.dino_wm_reward_loss.DINOWMRewardLoss` against the
	task's recorded rewards (`train_tc_wm_reward_*.yaml`) and then loaded frozen beside this
	model at evaluation time (`frozen_tc_wm_reward_model_*.yaml`). The two meet through
	`_unflatten_latent` and `write_action_into_newest_frame`, which are the only things the
	head asks of a world model — the same two on all four DINO-WM-family wrappers, which is
	why one head serves all of them.

	The practical consequence is that this model plans with an ordinary per-step reward, so
	`planner.cfg.use_value: false` is correct for it and `discount` trades intermediate reward
	against terminal value the way it does anywhere else. It also means the objective is the
	task's reward rather than TC-WM's latent distance, so planning numbers from here are not
	directly comparable with TC-WM's own reported ones.

	### Assumptions this wrapper makes about the task

	* The task's observation is a dict/TensorDict. `cfg.visual_key` names the RGB frames
	  and `cfg.proprio_keys` the vector observations that make up TC-WM's `proprio`;
	  their concatenated width must match what the WM's proprio encoder was trained on.
	* The task's frame stack (`task.cfg.num_frames`) equals the WM's `num_hist`, so one
	  observation from the task is exactly one predictor window.
	* `cfg.tcwm_cfg.frameskip > 1` means one WM step covers several env steps, and one WM
	  action is `frameskip` env actions concatenated. There are two ways a task can meet
	  that, and `_validate_task_compatibility` picks between them by action width:

	  - The task's own action *is* the concatenated chunk, because its env applies
	    `frame_skip` primitive actions per step (`custom_maniskill_tasks.FrameSkip`, as
	    `PushTTask` and `ManiSkillTask` both do) and its dataset concatenates the same
	    ones. One WM step is then exactly one env step, and the planner searches the space
	    the WM was trained on. This is what a checkpoint trained at `frameskip > 1` needs
	    in order to be run faithfully — set `task.cfg.frame_skip` to the same value.
	  - The task's action is a single primitive action, and `_to_tcwm_action` tiles it
	    over the skip. A held-action approximation: it can only express a constant action
	    per WM step, and the WM's latent advances `frameskip` primitive steps while the env
	    advances one, so the two drift apart over an episode.

	### One process, one repo

	TC-WM and DINO-WM both import their own modules rootlessly under the same top-level
	names, so a single process can host one of them or the other, never both. Composing
	`frozen_tc_wm_*` and `frozen_dino_wm_*` into one config raises from
	`s2p.lib.tc_wm_path` rather than quietly building one model out of the other's classes.
	"""

	def __init__(self, cfg, task:OnlineTaskBase):
		super().__init__(cfg=cfg)
		self.cfg = cfg

		# This model's config node carries `_recursive_: false` — the only way to stop Hydra
		# instantiating the `_target_`s inside `cfg.tcwm_cfg`, which are TC-WM's own and
		# are Trainer's to build. That flag applies to every kwarg of the node, so `task`
		# arrives as a config here rather than as a built task. Rebuild it: task factories
		# memoize on the config hash, so this is the same instance every other model got, not
		# a second env.
		self.task = instantiate(task) if not isinstance(task, TaskBase) else task
		self.tcwm_config = self.cfg.tcwm_cfg

		with _root_log_handlers_preserved(), _wandb_suppressed():
			Trainer, default_transform = _import_tc_wm()
			trainer = Trainer(self.tcwm_config)
		self.tc_wm = _unwrap(trainer.model)
		for name in _TCWM_MODULE_KEYS:
			submodule = getattr(self.tc_wm, name, None)
			if submodule is not None:
				setattr(self.tc_wm, name, _unwrap(submodule))
		self.tc_wm = self.tc_wm.to(self.cfg.device)

		if self.tc_wm.predictor is None:
			raise ValueError(
				"This TC-WM was built with has_predictor=false, so it has no dynamics. "
				"It cannot be used as a world model."
			)

		self.num_hist = self.tc_wm.num_hist
		self.concat_dim = self.tc_wm.concat_dim
		self.frameskip = self.tcwm_config.frameskip

		# `concat_dim: 0` (proprio and action as extra *tokens*) is DINO-WM's other layout,
		# and TC-WM inherited the branches for it without them surviving the change of
		# latent: `encode` feeds the projector a (patches+2, encoder_emb_dim) tensor as a
		# single argument, which no TC-WM projector accepts — `MLP` is built with
		# `in_features = visual_emb_dim + proprio_emb_dim` and would raise on the width —
		# and `separate_emb` unpacks three slices into two names. Nothing in conf/ selects
		# it (common.yaml pins `concat_dim: 1`), so a checkpoint that claims it is a
		# checkpoint whose latent this class cannot read.
		if self.concat_dim != 1:
			raise ValueError(
				f"This TC-WM was trained with concat_dim={self.concat_dim}. Only concat_dim=1 "
				"(proprio and action tiled into the feature dimension) is supported here, and "
				"it is what every TC-WM config sets — TC-WM's own concat_dim=0 path does not "
				"work with a non-identity projector."
			)

		self._derive_latent_geometry()

		# How the task's observation dict maps onto TC-WM's (visual, proprio) pair.
		self.visual_key = self.cfg.visual_key
		self.dino_feature_key = self.cfg.dino_feature_key
		self.proprio_keys = list(self.cfg.proprio_keys)

		# Live env frames arrive as raw uint8 CHW; the dataset put its frames through
		# `default_transform` (resize, centre crop, Normalize(0.5, 0.5)) after scaling to
		# [0, 1]. DinoV3Encoder.forward assumes that [-1, 1] convention, so reproduce it.
		self.image_transform = default_transform(self.tcwm_config.img_size)

		# Action/proprio normalization stats. TC-WM normalizes both against dataset
		# statistics before they ever reach the model, so anything coming from the task
		# (raw env actions, raw proprio) has to be put through the same map or the
		# predictor sees out-of-distribution inputs. Proprio doubly so here: it is both an
		# encoder input and, in raw form, a block of latent channels the predictor predicts.
		self._register_normalization_stats(trainer)

		# `Trainer.init_models` resumes from `<resume_folder>/checkpoints/model_latest.pth`,
		# with that filename written out literally (TC-WM/train.py:739) — so a run whose best
		# epoch is not its last cannot be reached through `resume_folder` alone.
		# `cfg.resume_checkpoint` names the file to load over it, usually `model_best.pth`.
		#
		# Loaded through this class's own `load_from_file`, which understands the native
		# dict-of-modules format `Trainer.save_ckpt` writes — so this is the same load the
		# Trainer would have done, off a different file, and it re-derives the latent
		# geometry afterwards in case the swapped-in predictor carries a different token grid.
		# Before `_validate_task_compatibility`, which reads that geometry.
		self.resume_checkpoint = resolve_resume_checkpoint(
			self.cfg.resume_checkpoint, self.tcwm_config.resume_folder, type(self).__name__
		)
		if self.resume_checkpoint is not None:
			self.load_from_file(self.resume_checkpoint)

		self._validate_task_compatibility()

	# ------------------------------------------------------------------ setup helpers

	def _derive_latent_geometry(self) -> None:
		"""
		Read the token grid off the predictor's position embedding, and the three blocks a
		token splits into off the model.

		The position embedding is the one place the geometry is unambiguous: `train.py`
		derives `num_patches` from `img_size` and the encoder's patch size, and
		`predictor_dim` from `projected_dim + action_dim + raw_proprio_dim`, when it builds
		the ViT, and bakes the result into `pos_embedding` of shape
		`(1, num_hist * tokens_per_frame, token_dim)`. Recomputing that chain here would be
		a second, silently divergent copy of the same rule.

		The block widths come from the model's own attributes rather than from the config,
		because they are what `separate_s_a_p` and `replace_actions_from_z` slice by — so
		reading them anywhere else would risk this class and TC-WM disagreeing about where
		the action lives. The sum has to come back to the token width, which is the check
		below.
		"""
		pos_embedding = self.tc_wm.predictor.pos_embedding
		self.tokens_per_frame = pos_embedding.shape[1] // self.num_hist
		self.token_dim = pos_embedding.shape[2]
		self.latent_dim = self.num_hist * self.tokens_per_frame * self.token_dim

		self.projected_dim = self.tc_wm.projected_dim
		self.action_dim = self.tc_wm.action_dim
		self.raw_proprio_dim = self.tc_wm.raw_proprio_dim
		if self.projected_dim + self.action_dim + self.raw_proprio_dim != self.token_dim:
			raise ValueError(
				f"This TC-WM's predictor takes {self.token_dim}-wide tokens, but the model "
				f"splits a token into projected={self.projected_dim} + "
				f"action={self.action_dim} + proprio={self.raw_proprio_dim} = "
				f"{self.projected_dim + self.action_dim + self.raw_proprio_dim}. Every slice "
				"in this class, and in TC-WM's own `separate_s_a_p` and "
				"`replace_actions_from_z`, is keyed off `model.projected_dim`, so a "
				"disagreement means the action channels are read from the wrong offset. The "
				"usual cause is `projector: identity` with `projected_dim` left at TC-WM's "
				"default: conf/train_dinowm.yaml sets `projected_dim: ${encoder_emb_dim}` "
				"for exactly this reason."
			)

		# Width of one frame's action slot, in the same units `_action_embeddings` returns:
		# the un-tiled action embedding, before `num_action_repeat` copies of it fill the
		# `action_dim` channels.
		self.action_slot_dim = self.action_dim // self.tc_wm.num_action_repeat

	def _register_normalization_stats(self, trainer) -> None:
		# TrajSubset forwards attribute lookups to the underlying PushBlockDataset, which
		# is where the stats live (they are ones/zeros when normalize_action is false).
		dataset = trainer.train_traj_dset
		expected_action_dim = self.tc_wm.action_encoder.in_chans
		primitive_action_dim = max(1, expected_action_dim // self.frameskip)
		proprio_dim = self.tc_wm.proprio_encoder.in_chans

		def stat(name, default):
			value = getattr(dataset, name, None)
			return default if value is None else torch.as_tensor(value, dtype=torch.float32)

		# The action statistics on a TC-WM trajectory dataset are per *primitive* action:
		# it normalizes each one and only then concatenates `frameskip` of them
		# (TrajSlicerDataset). Tiling them here to the width of a world model action makes
		# that the same as normalizing the concatenated action in one go, which is what
		# `_to_tcwm_action` then does — whichever way the action reached full width.
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
				"TC-WM needs both an image and a proprioceptive vector, so the task must "
				f"expose a dict observation. Got {type(observation_dimension).__name__}."
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
		expected_proprio_dim = self.tc_wm.proprio_encoder.in_chans
		if proprio_dim != expected_proprio_dim:
			raise ValueError(
				f"cfg.proprio_keys {self.proprio_keys} give a {proprio_dim}-dim proprio "
				f"vector, but this world model's proprio encoder was trained on "
				f"{expected_proprio_dim} dims. The TC-WM PushCube dataset concatenates "
				"obs/agent/qpos, obs/agent/qvel and obs/extra/tcp_pose (9+9+7=25); either "
				"expose the same fields through the task's dataset_structure, or retrain "
				"the world model on the fields the task does expose."
			)

		# Which proprio a latent's trailing block holds depends on the projector, because
		# `VWorldModel.encode` writes a different thing in each case: the raw (normalized)
		# vector when the projector consumed the proprio embedding, and the tiled embedding
		# itself when the projector is an identity and consumed nothing. `_assemble_latent`
		# mirrors that, so the two widths have to agree or the concatenation is meaningless
		# even where it happens to fit.
		proprio_slot_dim = (
			self.tc_wm.proprio_encoder.emb_dim * self.tc_wm.num_proprio_repeat
			if self.tc_wm.identity_projector
			else expected_proprio_dim
		)
		if proprio_slot_dim != self.raw_proprio_dim:
			raise ValueError(
				f"This TC-WM's latent reserves {self.raw_proprio_dim} channels for proprio, "
				f"but with {'an identity' if self.tc_wm.identity_projector else 'a learned'} "
				f"projector `VWorldModel.encode` writes {proprio_slot_dim} there "
				f"({'proprio_emb_dim x num_proprio_repeat' if self.tc_wm.identity_projector else 'the raw proprio vector'}). "
				"The checkpoint and the config it was rebuilt from disagree."
			)

		# Either the task already acts in chunks of `frameskip` primitive actions, in which
		# case its action goes to the world model as it is, or it acts one primitive action
		# at a time and `_to_tcwm_action` tiles it. See the class docstring for why the
		# first is the one a checkpoint trained at frameskip > 1 needs. At frameskip 1 the
		# two are the same thing.
		expected_action_dim = self.tc_wm.action_encoder.in_chans
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

	# ------------------------------------------------------------- latent (un)flattening

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
		"""(b, t, 3, H, W) raw frames -> the [-1, 1] convention TC-WM's encoder expects."""
		b, t = visual.shape[:2]
		x = rearrange(visual, "b t c h w -> (b t) c h w")
		if not torch.is_floating_point(x):
			x = x.float() / 255.0
		x = self.image_transform(x)
		return rearrange(x, "(b t) c h w -> b t c h w", b=b)

	def _to_tcwm_obs(self, observation) -> typing.Tuple[dict, torch.Size]:
		"""
		Adapt one of this package's observations to TC-WM's obs dict.

		Observations here are `(*batch_dims, S, *feature_dims)` with the frame stack `S`
		innermost and `batch_dims` typically `(T, B)`; TC-WM wants `(b, t, ...)` with a
		single leading batch dim. Since `S == num_hist`, the mapping is just "collapse the
		batch dims and let the frame stack be TC-WM's time axis".

		The returned `proprio` is normalized and used twice downstream: `encode_obs` embeds
		it for the projector, and `_assemble_latent` writes it verbatim into the latent's
		proprio channels.
		"""
		try:
			keys = set(observation.keys())
		except AttributeError:
			raise ValueError(
				"TC-WM needs both an image and a proprioceptive vector, so `encode` must "
				f"be given a dict observation. Got {type(observation).__name__}."
			)

		tcwm_obs = {}
		# The dataset ships DINOv3 patch features precomputed with the very backbone this
		# model's encoder holds, and `encode_visual` short-circuits to them when present —
		# that is the whole reason offline training never runs the encoder. They are absent
		# from live env observations, which fall through to the visual path below.
		if self.dino_feature_key is not None and self.dino_feature_key in keys:
			features = observation[self.dino_feature_key]      # (*batch, S, P, D)
			batch_dims = features.shape[:-3]
			tcwm_obs["dino_patch_features"] = features.reshape(-1, *features.shape[-3:]).float()
		else:
			visual = observation[self.visual_key]              # (*batch, S, 3, H, W)
			batch_dims = visual.shape[:-4]
			tcwm_obs["visual"] = self._prepare_visual(visual.reshape(-1, *visual.shape[-4:]))

		proprio = torch.cat([observation[key] for key in self.proprio_keys], dim=-1)
		proprio = proprio.reshape(-1, *proprio.shape[-2:]).float()
		tcwm_obs["proprio"] = (proprio - self.proprio_mean) / self.proprio_std

		return tcwm_obs, batch_dims

	def _to_tcwm_action(self, action:torch.Tensor) -> torch.Tensor:
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

	def _assemble_latent(
			self, z_obs:dict, proprio:torch.Tensor, action_embedding:torch.Tensor
		) -> torch.Tensor:
		"""
		Mirror of `VWorldModel.encode`'s concat_dim=1 branch, but taking an already-embedded
		action.

		`encode` embeds actions itself, which is no use here: the action that belongs in
		frame `i`'s slot is usually not one we hold in raw form (it comes out of the
		previous latent), and the newest one has to be tiled the same way regardless.

		The two projector cases are TC-WM's own and differ in what the trailing block holds
		— see `_validate_task_compatibility`. `proprio` is the normalized raw vector, i.e.
		`obs["proprio"]` as `encode` reads it.
		"""
		num_patches = z_obs["visual"].shape[2]
		proprio_tiled = repeat(z_obs["proprio"].unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.tc_wm.num_proprio_repeat)
		action_tiled = repeat(action_embedding.unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		action_repeated = action_tiled.repeat(1, 1, 1, self.tc_wm.num_action_repeat)

		if self.tc_wm.identity_projector:
			return torch.cat([z_obs["visual"], action_repeated, proprio_repeated], dim=3)

		projected = self.tc_wm.post_concat_projection(z_obs["visual"], proprio_repeated)
		proprio_raw_tiled = repeat(proprio.unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		return torch.cat([projected, action_repeated, proprio_raw_tiled], dim=3)

	def _zero_action_embedding(self, batch_size:int, num_frames:int, device) -> torch.Tensor:
		"""
		`encode_act` of the zero action — what an unknown or not-yet-chosen slot holds.

		Writing literal zeros there instead would be out of distribution: `encode_act` is a
		1x1 convolution *with a bias*, so no action embeds to the zero vector and the
		predictor has never seen one. The difference is the bias, which is small in
		magnitude but sits in every one of the tokens' action channels, and at the start of
		an episode it is two of the three frames the predictor attends over.

		The zero action is normalized the same way a real one is (`_to_tcwm_action`), so
		this is genuinely "the agent took the null action", not "these channels are blank".
		For a delta action space that reads as "nothing was commanded to move", which is the
		honest stand-in for a step that never happened.

		Returns `(batch_size, num_frames, action_slot_dim)`, matching `_action_slots`.
		"""
		zero = torch.zeros(
			batch_size, num_frames, self.tc_wm.action_encoder.in_chans, device=device
		)
		return self.tc_wm.encode_act((zero - self.action_mean) / self.action_std)

	def _clear_action_slot(self, z:torch.Tensor) -> torch.Tensor:
		"""
		Reset the action slot of every frame of `z` to this class's "unset" marker.

		The marker is `encode_act(0)` rather than a literal zero, for the reason
		`_zero_action_embedding` gives, and it is written through TC-WM's own
		`replace_actions_from_z` so that the offset comes from the one implementation the
		rest of TC-WM agrees with. Returns a copy, since the callers are holding latents the
		planners still own.

		In practice nothing reads a slot left in this state — `dynamics` overwrites the
		newest frame before predicting, and `_action_slots` drops it — but keeping it equal
		to what `_action_slots` writes is what makes a rolled-out latent and an encoded one
		interchangeable slot-for-slot.
		"""
		batch_size, num_frames = z.shape[:2]
		zero = torch.zeros(
			batch_size, num_frames, self.tc_wm.action_encoder.in_chans, device=z.device
		)
		return self.tc_wm.replace_actions_from_z(
			z.clone(), (zero - self.action_mean) / self.action_std
		)

	def _action_embeddings(self, z:torch.Tensor) -> torch.Tensor:
		"""
		The de-tiled action embeddings of a latent window: `(b, t, action_slot_dim)`.

		`separate_s_a_p` is the accessor to use rather than `separate_emb`: it is what
		`predict`, `rollout` and `replace_actions_from_z` are written against, whereas
		`separate_emb` splits at `proprio_dim` (the *embedded* proprio width) and so
		describes DINO-WM's layout, not this one — and returns no action at all.

		The action block holds `num_action_repeat` copies of one embedding, tiled across
		every patch, so one patch and one copy is the whole content.
		"""
		action = self.tc_wm.separate_s_a_p(z)["action"]
		return action[:, :, 0, :self.action_slot_dim]

	def _action_slots(self, previous_state, action, batch_size:int, device) -> torch.Tensor:
		"""
		Build the per-frame action embeddings for a freshly encoded window.

		Frame `i` of a TC-WM window carries the action applied *at* frame `i`, taking it
		to frame `i+1`. That is the alignment the model was trained on — the recording puts
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
		"""
		# Nothing is known: an offline encoding that was handed no action at all, or a
		# window whose only slot is the unset one (`num_hist == 1`, where `action` was
		# applied at a frame that is no longer in the window).
		if action is None or self.num_hist == 1:
			return self._zero_action_embedding(batch_size, self.num_hist, device)

		unset = self._zero_action_embedding(batch_size, 1, device)
		applied = self.tc_wm.encode_act(self._to_tcwm_action(action))   # (b, 1, E)
		if previous_state is None:
			# `num_hist == 2` leaves nothing older than `applied`; the convolution would
			# handle a zero-length sequence, but an explicit empty tensor is clearer.
			older = (
				self._zero_action_embedding(batch_size, self.num_hist - 2, device)
				if self.num_hist > 2
				else torch.zeros(batch_size, 0, self.action_slot_dim, device=device)
			)
		else:
			older = self._action_embeddings(self._unflatten_latent(previous_state))[:, 1:-1]
		return torch.cat([older, applied, unset], dim=1)

	# ------------------------------------------------------------------ encoder model

	def encode(self, observation, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
		tcwm_obs, batch_dims = self._to_tcwm_obs(observation)
		z_obs = self.tc_wm.encode_obs(tcwm_obs)

		batch_size, num_frames = z_obs["proprio"].shape[:2]
		if num_frames != self.num_hist:
			raise ValueError(
				f"Observation carries {num_frames} stacked frames, but the world model's "
				f"predictor window is {self.num_hist} frames wide."
			)

		action_embedding = self._action_slots(
			previous_state, action, batch_size, z_obs["proprio"].device
		)
		z = self._assemble_latent(z_obs, tcwm_obs["proprio"], action_embedding)
		return self._flatten_latent(z, batch_dims)

	def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		encoder_module = _TCWMEncoderForFLOPS(
			encoder=self.tc_wm.encoder,
			proprio_encoder=self.tc_wm.proprio_encoder,
			action_encoder=self.tc_wm.action_encoder,
			post_concat_projection=self.tc_wm.post_concat_projection,
			encoder_transform=self.tc_wm.encoder_transform,
			identity_projector=self.tc_wm.identity_projector,
			num_proprio_repeat=self.tc_wm.num_proprio_repeat,
		)
		return (
			(self.num_hist, 3, self.tcwm_config.img_size, self.tcwm_config.img_size),
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
		serves every DINO-WM-family wrapper. `tc_wm` and `_to_tcwm_action` are what
		differ between them.

		Takes an unflattened window `(batch, num_frames, tokens, dim)` and returns one, leaving
		the caller's `z` untouched: TC-WM's `replace_actions_from_z` writes in place, so the
		frame it is handed is cloned first.
		"""
		newest = self.tc_wm.replace_actions_from_z(
			z[:, -1:].clone(), self._to_tcwm_action(action)
		)
		return torch.cat([z[:, :-1], newest], dim=1)

	def dynamics(self, s:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
		batch_dims = s.shape[:-1]
		z = self._unflatten_latent(s).clone()

		# Write the action into the newest frame's slot, predict, and slide the window by
		# one — the same three steps as one iteration of `VWorldModel.rollout`. The clone
		# above matters: `replace_actions_from_z` writes in place, and the planners reuse
		# the latent they pass in (MPPI scores the reward head off it after this returns).
		z = self.write_action_into_newest_frame(z, a)

		# `predict` returns the predictor's auxiliary losses alongside the latent — an empty
		# dict for the deterministic ViT, a KL for the RSSM variant. Neither is a planning
		# quantity, so it is dropped here.
		z_predicted, _ = self.tc_wm.predict(z)

		# The predictor emits a full token, action channels included, for the frame it just
		# predicted — but the action applied *at* that frame is the next one to be chosen,
		# so those channels are cleared rather than kept. TC-WM does the same by writing
		# the plan's next action over them (`rollout`), and the next `dynamics` call does
		# exactly that; clearing them in the meantime is what keeps a rolled-out latent
		# interchangeable with an encoded one, which MPPI relies on when it scores the
		# reward head off whichever of the two it is holding.
		z_new = self._clear_action_slot(z_predicted[:, -1:])
		# `rollout` refreshes any fixed special tokens on each newly predicted frame; a
		# no-op for VWorldModel, and kept so a subclass that needs it is not silently
		# skipped here.
		z_new = self.tc_wm.refresh_special_tokens(z_new)
		z_next = torch.cat([z[:, 1:], z_new], dim=1)
		return self._flatten_latent(z_next, batch_dims)

	def get_dynamics_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		dynamics_module = _TCWMDynamicsForFLOPS(
			predictor=self.tc_wm.predictor,
			action_encoder=self.tc_wm.action_encoder,
			latent_dim=self.latent_dim,
			num_hist=self.num_hist,
			tokens_per_frame=self.tokens_per_frame,
			token_dim=self.token_dim,
		)
		return (
			(self.latent_dim + self.tc_wm.action_encoder.in_chans,),
			dynamics_module,
		)

	# ------------------------------------------------------------------------ extras

	def decode(self, s:torch.Tensor) -> torch.Tensor:
		"""
		Render a latent window back to images, for inspecting rollouts. Returns
		`(*batch_dims, num_hist, 3, H, W)`; requires the model to have a decoder.

		Two stages, which is TC-WM's own decode path (`train.py`'s `openloop_rollout` and
		`rollout.py`): `emb_decoder` lifts the projected latent back into the backbone's
		embedding space, and `decoder` turns that into pixels. Note that
		`VWorldModel.decode` cannot be called on a whole latent — it forwards its argument
		straight to the decoder as if it were already an embedding, so it has to be handed
		the projected block, after `emb_decoder`, rather than the full token.
		"""
		if self.tc_wm.decoder is None:
			raise ValueError("This TC-WM was built with has_decoder=false.")
		batch_dims = s.shape[:-1]
		projected = self.tc_wm.separate_s_a_p(self._unflatten_latent(s))["projected"]
		obs, _ = self.tc_wm.decode(self.tc_wm.emb_decoder(projected))
		visual = obs["visual"]
		return visual.reshape(*batch_dims, *visual.shape[1:])

	def train(self, mode:bool=True):
		"""
		Keep a frozen world model in eval mode whatever mode the agent is put in.

		The ViT predictor carries dropout (0.1 in conf/predictor/vit.yaml) and TC-WM's
		projector another (conf/projector/mlp.yaml), so a world model left in train mode
		returns a *different* rollout for the same state and action on every call — noise
		injected straight into every planner score, and a different function from the one
		every evaluator sees under `model.eval()`.
		"""
		super().train(mode)
		if self.cfg.freeze:
			self.tc_wm.eval()
		return self

	def requires_grad_(self, requires_grad):
		return super().requires_grad_(requires_grad and not self.cfg.freeze)

	def save_to_file(self, filepath) -> None:
		# Frozen: these weights are still exactly the file the vendored Trainer resumed
		# from, so point at it instead of copying it. See s2p.lib.checkpointing.
		# Whichever file the weights actually came from: the one `cfg.resume_checkpoint`
		# named, or the Trainer's own resume when it named none.
		if not self.cfg.freeze:
			source = None
		elif self.resume_checkpoint is not None:
			source = self.resume_checkpoint
		else:
			source = resume_folder_checkpoint(self.tcwm_config.resume_folder)
		if source is not None:
			save_checkpoint_reference(filepath, source, type(self).__name__)
			return

		torch.save(self.state_dict(), filepath)

	def load_from_file(self, filepath:str) -> None:
		# May be a reference written by `save_to_file` above rather than weights.
		filepath = resolve_checkpoint_path(filepath)
		checkpoint = torch.load(filepath, map_location=self.cfg.device, weights_only=False)

		# A native TC-WM checkpoint is a dict of whole pickled modules keyed by role
		# (Trainer.save_ckpt / plan.load_ckpt), not a state dict — swap them in wholesale.
		# Anything else is assumed to be one of ours, written by `save_to_file`.
		is_native = isinstance(checkpoint, dict) and any(
			isinstance(checkpoint.get(key), torch.nn.Module) for key in _TCWM_MODULE_KEYS
		)
		if not is_native:
			self.load_state_dict(checkpoint)
			return

		for key in _TCWM_MODULE_KEYS:
			module = checkpoint.get(key)
			if isinstance(module, torch.nn.Module):
				setattr(self.tc_wm, key, _unwrap(module).to(self.cfg.device))
		# A swapped-in projector changes which of `_assemble_latent`'s two branches applies,
		# and a swapped-in predictor can carry a different token grid than the one built at
		# construction — every reshape in this class is keyed off it.
		self.tc_wm.identity_projector = isinstance(
			self.tc_wm.post_concat_projection, torch.nn.Identity
		)
		self._derive_latent_geometry()


class _TCWMEncoderForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats TC-WM's encoding of one observation window. Used
	for counting the FLOPs of the encoding path only.

	Note this measures the *live* path, in which frames go through the DINOv3 backbone.
	Offline batches carry precomputed `dino_patch_features` and skip the backbone entirely,
	so their true encoding cost is the proprio and action embeddings plus the projection.
	"""
	def __init__(
			self, encoder, proprio_encoder, action_encoder, post_concat_projection,
			encoder_transform, identity_projector, num_proprio_repeat
		):
		super().__init__()
		self.encoder = encoder
		self.proprio_encoder = proprio_encoder
		self.action_encoder = action_encoder
		self.post_concat_projection = post_concat_projection
		self.encoder_transform = encoder_transform
		self.identity_projector = identity_projector
		self.num_proprio_repeat = num_proprio_repeat

	def forward(self, visual:torch.Tensor):
		b, t = visual.shape[:2]
		x = rearrange(visual, "b t c h w -> (b t) c h w")
		visual_embedding = self.encoder(self.encoder_transform(x))
		visual_embedding = rearrange(visual_embedding, "(b t) p d -> b t p d", b=b)

		# Values do not affect FLOPs; only the shapes the two 1-D convolutions and the
		# projector see do.
		device = visual.device
		proprio_embedding = self.proprio_encoder(
			torch.zeros(b, t, self.proprio_encoder.in_chans, device=device)
		)
		self.action_encoder(torch.zeros(b, t, self.action_encoder.in_chans, device=device))
		if self.identity_projector:
			return visual_embedding

		num_patches = visual_embedding.shape[2]
		proprio_tiled = repeat(proprio_embedding.unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
		return self.post_concat_projection(visual_embedding, proprio_repeated)


class _TCWMDynamicsForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats one TC-WM prediction step from a flat
	(latent, action) input. Used for counting the FLOPs of the dynamics only.
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

def get_or_create(cfg, task) -> "TCWorldModel":
	key = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = TCWorldModel(cfg=cfg, task=task)
	return _registry[key]
