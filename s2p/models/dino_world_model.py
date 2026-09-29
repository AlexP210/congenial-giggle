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
from s2p.lib.dino_wm_path import ensure_dino_wm_importable

ensure_dino_wm_importable()

from dino_wm.train import Trainer
from dino_wm.datasets.img_transforms import default_transform
from s2p.lib.checkpointing import (
	resolve_checkpoint_path,
	resolve_resume_checkpoint,
	resume_folder_checkpoint,
	save_checkpoint_reference,
)


# Submodules that a native DINO-WM checkpoint (Trainer.save_ckpt) stores as whole pickled
# nn.Modules rather than as state dicts, mirroring plan.py's ALL_MODEL_KEYS.
_DINOWM_MODULE_KEYS = ("encoder", "predictor", "decoder", "proprio_encoder", "action_encoder")


@contextlib.contextmanager
def _wandb_suppressed():
	"""
	Keep DINO-WM's Trainer from claiming this process's wandb run.

	`Trainer.__init__` calls `wandb.init(project="dino_wm")` unconditionally and then renames
	the run after the world model. wandb's `reinit` default is "return the previous run", so
	every later init — including the one EvaluationRunner makes into the S2P project — hands
	back that first run, and every S2P metric is logged into `dino_wm` under the world
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


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
	"""
	Peel `torch.compile` / DDP wrappers off a module.

	Trainer ends `init_models` with `torch.compile(self.model)` and runs every submodule
	through `accelerator.prepare`, so the object it hands back may be an OptimizedModule
	(or a DistributedDataParallel) around the real VWorldModel. Attribute access proxies
	through those wrappers, but `state_dict` keys do not — they gain a `_orig_mod.` /
	`module.` prefix, which would make every checkpoint written here unloadable by the
	original DINO-WM code. Unwrap once, up front, so both stay in the same namespace.
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


class DINOWorldModel(
	EncoderModelBase,
	DynamicsModelBase,
	# PolicyModelBase,       # DINO-WM has no policy
	torch.nn.Module
):
	"""
	DINO-WM wrapper, exposing a trained `VWorldModel` through the encoder and dynamics
	interfaces the planners and losses in this package expect.

	### What "state" means here

	DINO-WM is not Markov in a single frame: its predictor is a ViT over
	`num_hist * num_patches` tokens, so a one-step prediction needs the whole history
	window. The latent this wrapper hands out is therefore the *flattened window*

		s = flatten(z[:, -num_hist:])        # (num_hist * tokens_per_frame * token_dim,)

	where `z` is exactly DINO-WM's own latent — visual patch tokens with the proprio and
	action embeddings concatenated in (`concat_dim=1`) or appended as extra tokens
	(`concat_dim=0`). `dynamics` writes the incoming action into the newest frame's action
	slot, calls `predictor`, and slides the window by one, which is precisely what
	`VWorldModel.rollout` does.

	Frame `i`'s action slot holds the action applied *at* frame `i`, taking it to frame
	`i+1`, so the newest frame's action is one nothing has chosen yet. Every latent this
	class hands out — from `encode` and from `dynamics` alike — therefore carries zero
	there, and only `dynamics` fills it in, for the duration of one prediction. See
	`_action_slots`. That makes `s` Markov, at the cost of being large: for
	dinov3_vits16 at 224px with `concat_dim=1` and `num_hist=3` it is 3*196*404 ≈ 238k
	floats *per sample*, i.e. ~0.5 GB for an MPPI population of 512. Turn
	`planner.cfg.num_samples` down accordingly.

	### Where the reward comes from

	Nowhere in here. This is an encoder and a dynamics model, and nothing else: it implements
	neither `RewardModelBase` nor `StateValueModelBase`, so `AgentModel` cannot be handed it
	as `reward_model` or `value_model` and `MPPIPlanner` will never ask it to score anything.

	That is a deliberate narrowing of what DINO-WM's own code does. Upstream it plans by driving
	the latent towards a goal latent — `planning.objectives.create_objective_fn(mode="last")`,
	a terminal cost — which makes the model goal-conditioned: undefined until a goal is set, and
	re-aimed at every reset because the goal belongs to the episode rather than to the model.

	Instead the reward is a *separate* model fitted on top of this one:
	`s2p.models.dino_wm_reward_model.DINOWMRewardModel`, a cross-attention head over this
	latent window, trained by `s2p.losses.dino_wm_reward_loss.DINOWMRewardLoss` against the
	task's recorded rewards (`train_dino_wm_reward_*.yaml`) and then loaded frozen beside this
	model at evaluation time (`frozen_dino_wm_reward_model_*.yaml`). The two meet through
	`_unflatten_latent` and `write_action_into_newest_frame`, which are the only things the
	head asks of a world model — the same two on all four DINO-WM-family wrappers, which is
	why one head serves all of them.

	The practical consequence is that this model plans with an ordinary per-step reward, so
	`planner.cfg.use_value: false` is correct for it and `discount` trades intermediate reward
	against terminal value the way it does anywhere else. It also means the objective is the
	task's reward rather than DINO-WM's latent distance, so planning numbers from here are not
	directly comparable with DINO-WM's own reported ones.

	### Assumptions this wrapper makes about the task

	* The task's observation is a dict/TensorDict. `cfg.visual_key` names the RGB frames
	  and `cfg.proprio_keys` the vector observations that make up DINO-WM's `proprio`;
	  their concatenated width must match what the WM's proprio encoder was trained on.
	* The task's frame stack (`task.cfg.num_frames`) equals the WM's `num_hist`, so one
	  observation from the task is exactly one predictor window.
	* `cfg.dinowm_cfg.frameskip > 1` means one WM step covers several env steps, and one WM
	  action is `frameskip` env actions concatenated. There are two ways a task can meet
	  that, and `_validate_task_compatibility` picks between them by action width:

	  - The task's own action *is* the concatenated chunk, because its env applies
	    `frame_skip` primitive actions per step (`custom_maniskill_tasks.FrameSkip`, as
	    `PushTTask` and `ManiSkillTask` both do) and its dataset concatenates the same
	    ones. One WM step is then exactly one env step, and the planner searches the space
	    the WM was trained on. This is what a checkpoint trained at `frameskip > 1` needs
	    in order to be run faithfully — set `task.cfg.frame_skip` to the same value.
	  - The task's action is a single primitive action, and `_to_dinowm_action` tiles it
	    over the skip. A held-action approximation: it can only express a constant action
	    per WM step, and the WM's latent advances `frameskip` primitive steps while the env
	    advances one, so the two drift apart over an episode.
	"""

	def __init__(self, cfg, task:OnlineTaskBase):
		super().__init__(cfg=cfg)
		self.cfg = cfg

		# This model's config node carries `_recursive_: false` — the only way to stop Hydra
		# instantiating the `_target_`s inside `cfg.dinowm_cfg`, which are DINO-WM's own and
		# are Trainer's to build. That flag applies to every kwarg of the node, so `task`
		# arrives as a config here rather than as a built task. Rebuild it: task factories
		# memoize on the config hash, so this is the same instance every other model got, not
		# a second env.
		self.task = instantiate(task) if not isinstance(task, TaskBase) else task
		self.dinowm_config = self.cfg.dinowm_cfg

		with _wandb_suppressed():
			trainer = Trainer(self.dinowm_config)
		self.dino_wm = _unwrap(trainer.model)
		for name in _DINOWM_MODULE_KEYS:
			submodule = getattr(self.dino_wm, name, None)
			if submodule is not None:
				setattr(self.dino_wm, name, _unwrap(submodule))
		self.dino_wm = self.dino_wm.to(self.cfg.device)

		if self.dino_wm.predictor is None:
			raise ValueError(
				"This DINO-WM was built with has_predictor=false, so it has no dynamics. "
				"It cannot be used as a world model."
			)

		self.num_hist = self.dino_wm.num_hist
		self.concat_dim = self.dino_wm.concat_dim
		self.frameskip = self.dinowm_config.frameskip
		self._derive_latent_geometry()

		# How the task's observation dict maps onto DINO-WM's (visual, proprio) pair.
		self.visual_key = self.cfg.visual_key
		self.dino_feature_key = self.cfg.dino_feature_key
		self.proprio_keys = list(self.cfg.proprio_keys)

		# Live env frames arrive as raw uint8 CHW; the dataset put its frames through
		# `default_transform` (resize, centre crop, Normalize(0.5, 0.5)) after scaling to
		# [0, 1]. DinoV3Encoder.forward assumes that [-1, 1] convention, so reproduce it.
		self.image_transform = default_transform(self.dinowm_config.img_size)

		# Action/proprio normalization stats. DINO-WM normalizes both against dataset
		# statistics before they ever reach the model, so anything coming from the task
		# (raw env actions, raw proprio) has to be put through the same map or the
		# predictor sees out-of-distribution inputs.
		self._register_normalization_stats(trainer)

		# `Trainer.init_models` resumes from `<resume_folder>/checkpoints/model_latest.pth`,
		# with that filename written out literally (dino_wm/train.py:225) — so a run whose
		# best epoch is not its last cannot be reached through `resume_folder` alone.
		# `cfg.resume_checkpoint` names the file to load over it, usually `model_best.pth`.
		#
		# Loaded through this class's own `load_from_file`, which understands the native
		# dict-of-modules format `Trainer.save_ckpt` writes — so this is the same load the
		# Trainer would have done, off a different file, and it re-derives the latent
		# geometry afterwards in case the swapped-in predictor carries a different token grid.
		# Before `_validate_task_compatibility`, which reads that geometry.
		self.resume_checkpoint = resolve_resume_checkpoint(
			self.cfg.resume_checkpoint, self.dinowm_config.resume_folder, type(self).__name__
		)
		if self.resume_checkpoint is not None:
			self.load_from_file(self.resume_checkpoint)

		self._validate_task_compatibility()

	# ------------------------------------------------------------------ setup helpers

	def _derive_latent_geometry(self) -> None:
		"""
		Read the token grid off the predictor's position embedding.

		It is the one place the geometry is unambiguous: `train.py` derives `num_patches`
		from `img_size`, `concat_dim` and the encoder's `latent_ndim` when it builds the
		ViT, and bakes the result into `pos_embedding` of shape
		`(1, num_hist * tokens_per_frame, token_dim)`. Recomputing that chain here would
		be a second, silently divergent copy of the same rule.
		"""
		pos_embedding = self.dino_wm.predictor.pos_embedding
		self.tokens_per_frame = pos_embedding.shape[1] // self.num_hist
		self.token_dim = pos_embedding.shape[2]
		self.latent_dim = self.num_hist * self.tokens_per_frame * self.token_dim

		# Width of one frame's action slot, in the same units `separate_emb` returns:
		# a whole extra token for concat_dim=0, the un-tiled action embedding otherwise.
		if self.concat_dim == 0:
			self.action_slot_dim = self.token_dim
		else:
			self.action_slot_dim = self.dino_wm.action_dim // self.dino_wm.num_action_repeat

	def _register_normalization_stats(self, trainer:Trainer) -> None:
		# TrajSubset forwards attribute lookups to the underlying PushBlockDataset, which
		# is where the stats live (they are ones/zeros when normalize_action is false).
		dataset = trainer.train_traj_dset
		expected_action_dim = self.dino_wm.action_encoder.in_chans
		primitive_action_dim = max(1, expected_action_dim // self.frameskip)
		proprio_dim = self.dino_wm.proprio_encoder.in_chans

		def stat(name, default):
			value = getattr(dataset, name, None)
			return default if value is None else torch.as_tensor(value, dtype=torch.float32)

		# The action statistics on a DINO-WM trajectory dataset are per *primitive* action:
		# it normalizes each one and only then concatenates `frameskip` of them
		# (TrajSlicerDataset). Tiling them here to the width of a world model action makes
		# that the same as normalizing the concatenated action in one go, which is what
		# `_to_dinowm_action` then does — whichever way the action reached full width.
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
				"DINO-WM needs both an image and a proprioceptive vector, so the task must "
				f"expose a dict observation. Got {type(observation_dimension).__name__}."
			)

		# One task observation must be exactly one predictor window. Read the stack depth off
		# whichever visual leaf the task actually carries: an offline recording that ships
		# precomputed features has no reason to also carry the frames they were computed from
		# (and on a MIG partition, where the renderer will not start, the env cannot supply the
		# rgb dim either). Both leaves live under `obs` and are frame-stacked alike, so they
		# report the same depth wherever both exist.
		stack_key = self.visual_key if self.visual_key in observation_dimension else self.dino_feature_key
		if stack_key not in observation_dimension:
			raise ValueError(
				f"DINO-WM needs a frame-stacked visual leaf to size its predictor window, but the "
				f"task exposes neither {self.visual_key!r} nor {self.dino_feature_key!r}. Got "
				f"{sorted(observation_dimension)}."
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
		expected_proprio_dim = self.dino_wm.proprio_encoder.in_chans
		if proprio_dim != expected_proprio_dim:
			raise ValueError(
				f"cfg.proprio_keys {self.proprio_keys} give a {proprio_dim}-dim proprio "
				f"vector, but this world model's proprio encoder was trained on "
				f"{expected_proprio_dim} dims. The DINO-WM PushCube dataset concatenates "
				"obs/agent/qpos, obs/agent/qvel and obs/extra/tcp_pose (9+9+7=25); either "
				"expose the same fields through the task's dataset_structure, or retrain "
				"the world model on the fields the task does expose."
			)

		# Either the task already acts in chunks of `frameskip` primitive actions, in which
		# case its action goes to the world model as it is, or it acts one primitive action
		# at a time and `_to_dinowm_action` tiles it. See the class docstring for why the
		# first is the one a checkpoint trained at frameskip > 1 needs. At frameskip 1 the
		# two are the same thing.
		expected_action_dim = self.dino_wm.action_encoder.in_chans
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
		"""(b, t, 3, H, W) raw frames -> the [-1, 1] convention DINO-WM's encoder expects."""
		b, t = visual.shape[:2]
		x = rearrange(visual, "b t c h w -> (b t) c h w")
		if not torch.is_floating_point(x):
			x = x.float() / 255.0
		x = self.image_transform(x)
		return rearrange(x, "(b t) c h w -> b t c h w", b=b)

	def _to_dinowm_obs(self, observation) -> typing.Tuple[dict, torch.Size]:
		"""
		Adapt one of this package's observations to DINO-WM's obs dict.

		Observations here are `(*batch_dims, S, *feature_dims)` with the frame stack `S`
		innermost and `batch_dims` typically `(T, B)`; DINO-WM wants `(b, t, ...)` with a
		single leading batch dim. Since `S == num_hist`, the mapping is just "collapse the
		batch dims and let the frame stack be DINO-WM's time axis".
		"""
		try:
			keys = set(observation.keys())
		except AttributeError:
			raise ValueError(
				"DINO-WM needs both an image and a proprioceptive vector, so `encode` must "
				f"be given a dict observation. Got {type(observation).__name__}."
			)

		dinowm_obs = {}
		# The dataset ships DINOv3 patch features precomputed with the very backbone this
		# model's encoder holds, and `encode_obs` short-circuits to them when present —
		# that is the whole reason offline training never runs the encoder. They are absent
		# from live env observations, which fall through to the visual path below.
		if self.dino_feature_key is not None and self.dino_feature_key in keys:
			features = observation[self.dino_feature_key]      # (*batch, S, P, D)
			batch_dims = features.shape[:-3]
			dinowm_obs["dino_patch_features"] = features.reshape(-1, *features.shape[-3:]).float()
		else:
			visual = observation[self.visual_key]              # (*batch, S, 3, H, W)
			batch_dims = visual.shape[:-4]
			dinowm_obs["visual"] = self._prepare_visual(visual.reshape(-1, *visual.shape[-4:]))

		proprio = torch.cat([observation[key] for key in self.proprio_keys], dim=-1)
		proprio = proprio.reshape(-1, *proprio.shape[-2:]).float()
		dinowm_obs["proprio"] = (proprio - self.proprio_mean) / self.proprio_std

		return dinowm_obs, batch_dims

	def _to_dinowm_action(self, action:torch.Tensor) -> torch.Tensor:
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
		proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.dino_wm.num_proprio_repeat)
		action_tiled = repeat(action_embedding.unsqueeze(2), "b t 1 a -> b t f a", f=num_patches)
		action_repeated = action_tiled.repeat(1, 1, 1, self.dino_wm.num_action_repeat)
		return torch.cat([z_obs["visual"], proprio_repeated, action_repeated], dim=3)

	def _zero_action_embedding(self, batch_size:int, num_frames:int, device) -> torch.Tensor:
		"""
		`encode_act` of the zero action — what an unknown or not-yet-chosen slot holds.

		Writing literal zeros there instead would be out of distribution: `encode_act` is a
		1x1 convolution *with a bias*, so no action embeds to the zero vector and the
		predictor has never seen one. The difference is the bias, which is small in
		magnitude but sits in every one of the 588 tokens' trailing channels, and at the
		start of an episode it is two of the three frames the predictor attends over.

		The zero action is normalized the same way a real one is (`_to_dinowm_action`), so
		this is genuinely "the agent took the null action", not "these channels are blank".
		For PushT's relative action space that reads as "the target did not move", which is
		the honest stand-in for a step that never happened.

		Returns `(batch_size, num_frames, action_slot_dim)`, matching `_action_slots`.
		"""
		zero = torch.zeros(
			batch_size, num_frames, self.dino_wm.action_encoder.in_chans, device=device
		)
		return self.dino_wm.encode_act((zero - self.action_mean) / self.action_std)

	def _clear_action_slot(self, z:torch.Tensor) -> torch.Tensor:
		"""
		Reset the action slot of every frame of `z` to this class's "unset" marker.

		The marker is `encode_act(0)` rather than a literal zero, for the reason
		`_zero_action_embedding` gives, and it is written through DINO-WM's own
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
			batch_size, num_frames, self.dino_wm.action_encoder.in_chans, device=z.device
		)
		return self.dino_wm.replace_actions_from_z(
			z.clone(), (zero - self.action_mean) / self.action_std
		)

	def _action_slots(self, previous_state, action, batch_size:int, device) -> torch.Tensor:
		"""
		Build the per-frame action embeddings for a freshly encoded window.

		Frame `i` of a DINO-WM window carries the action applied *at* frame `i`, taking it
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
		applied = self.dino_wm.encode_act(self._to_dinowm_action(action))   # (b, 1, E)
		if previous_state is None:
			# `num_hist == 2` leaves nothing older than `applied`; the convolution would
			# handle a zero-length sequence, but an explicit empty tensor is clearer.
			older = (
				self._zero_action_embedding(batch_size, self.num_hist - 2, device)
				if self.num_hist > 2
				else torch.zeros(batch_size, 0, self.action_slot_dim, device=device)
			)
		else:
			_, previous_actions = self.dino_wm.separate_emb(self._unflatten_latent(previous_state))
			older = previous_actions[:, 1:-1]
		return torch.cat([older, applied, unset], dim=1)

	# ------------------------------------------------------------------ encoder model

	def encode(self, observation, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
		dinowm_obs, batch_dims = self._to_dinowm_obs(observation)
		z_obs = self.dino_wm.encode_obs(dinowm_obs)

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
		encoder_module = _DINOWMEncoderForFLOPS(
			encoder=self.dino_wm.encoder,
			proprio_encoder=self.dino_wm.proprio_encoder,
			action_encoder=self.dino_wm.action_encoder,
			encoder_transform=self.dino_wm.encoder_transform,
		)
		return (
			(self.num_hist, 3, self.dinowm_config.img_size, self.dinowm_config.img_size),
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
		serves every DINO-WM-family wrapper. `dino_wm` and `_to_dinowm_action` are what
		differ between them.

		Takes an unflattened window `(batch, num_frames, tokens, dim)` and returns one, leaving
		the caller's `z` untouched: DINO-WM's `replace_actions_from_z` writes in place, so the
		frame it is handed is cloned first.
		"""
		newest = self.dino_wm.replace_actions_from_z(
			z[:, -1:].clone(), self._to_dinowm_action(action)
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

		z_predicted = self.dino_wm.predict(z)

		# The predictor emits a full token, action channels included, for the frame it just
		# predicted — but the action applied *at* that frame is the next one to be chosen,
		# so those channels are cleared rather than kept. DINO-WM does the same by writing
		# the plan's next action over them (`rollout`), and the next `dynamics` call does
		# exactly that; clearing them in the meantime is what keeps a rolled-out latent
		# interchangeable with an encoded one, which MPPI relies on when it scores the
		# reward head off whichever of the two it is holding.
		z_new = self._clear_action_slot(z_predicted[:, -1:])
		z_next = torch.cat([z[:, 1:], z_new], dim=1)
		return self._flatten_latent(z_next, batch_dims)

	def get_dynamics_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		dynamics_module = _DINOWMDynamicsForFLOPS(
			predictor=self.dino_wm.predictor,
			action_encoder=self.dino_wm.action_encoder,
			latent_dim=self.latent_dim,
			num_hist=self.num_hist,
			tokens_per_frame=self.tokens_per_frame,
			token_dim=self.token_dim,
		)
		return (
			(self.latent_dim + self.dino_wm.action_encoder.in_chans,),
			dynamics_module,
		)

	# ------------------------------------------------------------------------ extras

	def decode(self, s:torch.Tensor) -> torch.Tensor:
		"""
		Render a latent window back to images, for inspecting rollouts. Returns
		`(*batch_dims, num_hist, 3, H, W)`; requires the model to have a decoder.
		"""
		if self.dino_wm.decoder is None:
			raise ValueError("This DINO-WM was built with has_decoder=false.")
		batch_dims = s.shape[:-1]
		obs, _ = self.dino_wm.decode(self._unflatten_latent(s))
		visual = obs["visual"]
		return visual.reshape(*batch_dims, *visual.shape[1:])

	def train(self, mode:bool=True):
		"""
		Keep a frozen world model in eval mode whatever mode the agent is put in.

		The ViT predictor carries dropout (0.1 in conf/predictor/vit.yaml), so a world
		model left in train mode returns a *different* rollout for the same state and
		action on every call — noise injected straight into every planner score, and a
		different function from the one every evaluator sees under `model.eval()`.
		"""
		super().train(mode)
		if self.cfg.freeze:
			self.dino_wm.eval()
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
			source = resume_folder_checkpoint(self.dinowm_config.resume_folder)
		if source is not None:
			save_checkpoint_reference(filepath, source, type(self).__name__)
			return

		torch.save(self.state_dict(), filepath)

	def load_from_file(self, filepath:str) -> None:
		# May be a reference written by `save_to_file` above rather than weights.
		filepath = resolve_checkpoint_path(filepath)
		checkpoint = torch.load(filepath, map_location=self.cfg.device, weights_only=False)

		# A native DINO-WM checkpoint is a dict of whole pickled modules keyed by role
		# (Trainer.save_ckpt / plan.load_ckpt), not a state dict — swap them in wholesale.
		# Anything else is assumed to be one of ours, written by `save_to_file`.
		is_native = isinstance(checkpoint, dict) and any(
			isinstance(checkpoint.get(key), torch.nn.Module) for key in _DINOWM_MODULE_KEYS
		)
		if not is_native:
			self.load_state_dict(checkpoint)
			return

		for key in _DINOWM_MODULE_KEYS:
			module = checkpoint.get(key)
			if isinstance(module, torch.nn.Module):
				setattr(self.dino_wm, key, _unwrap(module).to(self.cfg.device))
		# A swapped-in predictor can carry a different token grid than the one built at
		# construction, and every reshape in this class is keyed off it.
		self._derive_latent_geometry()


class _DINOWMEncoderForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats DINO-WM's encoding of one observation window. Used
	for counting the FLOPs of the encoding path only.

	Note this measures the *live* path, in which frames go through the DINOv3 backbone.
	Offline batches carry precomputed `dino_patch_features` and skip the backbone entirely,
	so their true encoding cost is only the proprio and action embeddings.
	"""
	def __init__(self, encoder, proprio_encoder, action_encoder, encoder_transform):
		super().__init__()
		self.encoder = encoder
		self.proprio_encoder = proprio_encoder
		self.action_encoder = action_encoder
		self.encoder_transform = encoder_transform

	def forward(self, visual:torch.Tensor):
		b, t = visual.shape[:2]
		x = rearrange(visual, "b t c h w -> (b t) c h w")
		visual_embedding = self.encoder(self.encoder_transform(x))
		visual_embedding = rearrange(visual_embedding, "(b t) p d -> b t p d", b=b)

		# Values do not affect FLOPs; only the shapes the two 1-D convolutions see do.
		device = visual.device
		self.proprio_encoder(torch.zeros(b, t, self.proprio_encoder.in_chans, device=device))
		self.action_encoder(torch.zeros(b, t, self.action_encoder.in_chans, device=device))
		return visual_embedding


class _DINOWMDynamicsForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats one DINO-WM prediction step from a flat
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

def get_or_create(cfg, task) -> "DINOWorldModel":
	key = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = DINOWorldModel(cfg=cfg, task=task)
	return _registry[key]
