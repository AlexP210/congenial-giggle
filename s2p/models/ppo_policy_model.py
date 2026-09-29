"""
Wrap a task-specific visual PPO expert as a policy this package can act with.

The experts are the product of `tools/ppo_visual_expert_fast.py`: one NatureCNN over the
camera image plus a linear embedding of a proprioceptive state vector, feeding a shared
latent to an actor and a critic. A run leaves `best_ckpt.pt`/`final_ckpt.pt` behind, e.g.

    /data/.../maniskill/PushCube-v1.1-ppo-visual-wrist-224-pd_ee_delta_pos/final_ckpt.pt

and `training_summary.json` beside them records which camera, resolution and control mode
the expert was trained through -- all three have to match the task this is handed, since
the expert has no way to tell a different view from a changed scene.

Those checkpoints are bare `state_dict`s, not pickled modules, so the architecture has to
be rebuilt here before the weights can go in. `_NatureCNN`/`_PPOExpert` below mirror
`ppo_visual_expert_fast.py`'s `NatureCNN`/`Agent` rather than importing them: that script
is a training entry point that pulls in tyro, wandb and tensorboard and sets a dynamo env
var at import, none of which belongs in a model constructor. The layer *shapes* are read
off the checkpoint instead of being hardcoded, so any expert from that script loads, and
`load_state_dict(strict=True)` is what catches the architecture drifting from the file.

### What this adapts between

The expert was trained on `FlattenRGBDObservationWrapper`'s two-key observation -- `rgb`
as `(N, H, W, C)` uint8 with every camera concatenated on the channel axis, and `state` as
ManiSkill's flattened non-sensor observation -- while an observation here is a TensorDict
of `(*batch_dims, S, *feature_dims)` leaves with the frame stack `S` innermost, one leaf
per nickname in the task's `dataset_structure`. So `policy` has to:

* **Take the newest frame.** The expert is memoryless: it was trained on single
  observations and there is no time axis in its input. A task that stacks frames for some
  other model's benefit (DINO-WM's `num_hist`, say) is fine -- `S > 1` is not an error,
  the older frames are simply not what this policy was trained to look at.
* **Rebuild `state` in the trained order.** `cfg.state_keys` is concatenated in sequence
  and must reproduce the order `flatten_state_dict` produced during training, which is
  insertion order over ManiSkill's observation dict -- `agent` before `extra`, and `qpos`
  before `qvel` within `agent`. For the PushCube expert above that is
  `[qpos, qvel, tcp_pose]` (9+9+7=25), the same concatenation
  `frozen_dino_wm_push_cube.yaml` documents for its own proprio encoder. Under an image
  obs mode the tasks withhold their privileged state, so this is proprioception and the
  tcp pose only, not a state-based shortcut around the camera.
* **Put the image back in NHWC.** `ManiSkillWrapper` permutes ManiSkill's frames to CHW on
  the way in; `_NatureCNN.forward` is kept exactly as trained and permutes CHW back out of
  NHWC itself, so the layout is undone here rather than the trained forward being edited.

The concatenated `state` width and the image's channel count and resolution are all
checked against the checkpoint at construction, because each of them is a silent failure
otherwise: a wrong key order or a differently-configured camera still produces a tensor of
the right rank, and the expert just acts badly on it.

### Actions

`policy` returns the deterministic action -- `actor_mean`, no sampling -- which is what an
eval rollout and an expert demo recording both use (`make_expert_demos.noisy_action` adds
its noise on top of exactly this). The actor's output is unbounded: ManiSkill's controller
clips to its own box when it applies an action, so the raw output and the effected action
diverge, which is why this project's recorded datasets hold actions well outside [-1, 1].
`cfg.clip_actions` clamps to `task.action_limits` so the returned action is the one the env
would actually execute; leave it off to see what the expert really proposed.
"""

import typing

import torch
import torch.nn as nn

from s2p.models.base.observation_policy_model_base import ObservationPolicyModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference


class _NatureCNN(nn.Module):
	"""
	The feature extractor of `ppo_visual_expert_fast.NatureCNN`, sized from a checkpoint.

	The conv stack is Nature-CNN's own (32/64/64 channels, 8/4/3 kernels, 4/2/1 strides) and
	fixed; everything that varies with how an expert was trained -- the channel count, the
	flattened width the resolution implies, the feature widths -- is passed in, read off the
	weights being loaded. `forward` is unchanged from the training script, including the
	NHWC->NCHW permute and the /255, so the expert sees what it saw while training.
	"""

	def __init__(self, in_channels, n_flatten, feature_size, state_dim, state_feature_size):
		super().__init__()
		extractors = {}
		self.out_features = 0

		cnn = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
		)
		extractors["rgb"] = nn.Sequential(
			cnn, nn.Linear(n_flatten, feature_size), nn.ReLU()
		)
		self.out_features += feature_size

		# `--no-include_state` trains an expert on pixels alone, and such a checkpoint has no
		# state extractor to size. Its absence is the signal, hence `None` rather than 0.
		if state_dim is not None:
			extractors["state"] = nn.Linear(state_dim, state_feature_size)
			self.out_features += state_feature_size

		self.extractors = nn.ModuleDict(extractors)

	def forward(self, observations) -> torch.Tensor:
		encoded_tensor_list = []
		for key, extractor in self.extractors.items():
			obs = observations[key]
			if key == "rgb":
				# NHWC uint8 as ManiSkill renders it -> NCHW float in [0, 1]
				obs = obs.float().permute(0, 3, 1, 2) / 255.0
			encoded_tensor_list.append(extractor(obs))
		return torch.cat(encoded_tensor_list, dim=1)


class _PPOExpert(nn.Module):
	"""
	`ppo_visual_expert_fast.Agent`, sized from a checkpoint instead of a sample observation.

	`critic` and `actor_logstd` are rebuilt although only `actor_mean` is ever called: they
	are in the checkpoint, and a `strict=True` load is the guard that this architecture still
	matches the file. Dropping them would trade that guard for two unexpected-key errors.

	The initial values do not matter and the training script's `layer_init` is not reproduced
	-- `PPOPolicyModel` requires a checkpoint, so every parameter here is overwritten before
	the module is ever called.
	"""

	def __init__(self, feature_net, hidden_size, n_act):
		super().__init__()
		self.feature_net = feature_net
		latent_size = self.feature_net.out_features
		self.critic = nn.Sequential(
			nn.Linear(latent_size, hidden_size),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_size, 1),
		)
		self.actor_mean = nn.Sequential(
			nn.Linear(latent_size, hidden_size),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_size, n_act),
		)
		self.actor_logstd = nn.Parameter(torch.zeros(1, n_act))

	def get_action(self, obs):
		"""The deterministic action, which is what an eval rollout and an expert rollout use."""
		return self.actor_mean(self.feature_net(obs))

	def forward(self, obs):
		return self.get_action(obs)


class PPOPolicyModel(ObservationPolicyModelBase, torch.nn.Module):

	def __init__(self, cfg, task:TaskBase):
		super().__init__(cfg)
		self.cfg = cfg
		self.task = task

		# How the task's observation dict maps onto the expert's (rgb, state) pair. Order
		# matters for `state_keys`: see the module docstring.
		self.visual_key = self.cfg.visual_key
		self.state_keys = None if self.cfg.state_keys is None else list(self.cfg.state_keys)

		# The expert is a whole trained policy rather than a layer sized by a config, so the
		# checkpoint is not optional the way a `fresh_*` model's is -- without it this is a
		# randomly initialised network that would still act, and score, and look like a
		# failed expert rather than a missing one.
		if self.cfg.checkpoint is None:
			raise ValueError(
				"PPOPolicyModel wraps a trained expert, so cfg.checkpoint is required. Point "
				"it at a ppo_visual_expert_fast.py checkpoint, e.g. "
				"<run folder>/best_ckpt.pt."
			)
		checkpoint_path = resolve_checkpoint_path(self.cfg.checkpoint)
		state_dict = torch.load(checkpoint_path, map_location="cpu")

		self.agent = self._build_agent(state_dict)
		self._validate_task_compatibility(state_dict)
		self.agent.load_state_dict(state_dict)
		self.agent.to(self.cfg.device)

		action_low, action_high = self.task.action_limits
		self.action_low = torch.as_tensor(action_low, dtype=torch.float, device=self.cfg.device)
		self.action_high = torch.as_tensor(action_high, dtype=torch.float, device=self.cfg.device)

		self.requires_grad_(True)

	# ------------------------------------------------------------------ construction

	def _build_agent(self, state_dict) -> _PPOExpert:
		"""
		The expert this checkpoint was saved from, its widths read off the weights.

		Reading them here rather than from the task is what makes the structural load below
		exact: the network is the checkpoint's by construction, and whether the *task* can
		feed it is a separate question, asked by `_validate_task_compatibility` so that it can
		fail with something more useful than a shape mismatch deep in a conv.
		"""
		try:
			in_channels = state_dict["feature_net.extractors.rgb.0.0.weight"].shape[1]
			n_flatten = state_dict["feature_net.extractors.rgb.1.weight"].shape[1]
			feature_size = state_dict["feature_net.extractors.rgb.1.weight"].shape[0]
			hidden_size, _ = state_dict["actor_mean.0.weight"].shape
			n_act = state_dict["actor_mean.2.weight"].shape[0]
		except KeyError as missing:
			raise ValueError(
				f"{self.cfg.checkpoint} is missing {missing}, so it is not a "
				"ppo_visual_expert_fast.py agent checkpoint. Note these files hold a bare "
				"state_dict of that script's `Agent`; an optimizer or trainer checkpoint "
				"will not load here."
			) from missing

		# Absent under `--no-include_state`, i.e. an expert trained on pixels alone.
		state_weight = state_dict.get("feature_net.extractors.state.weight")
		if state_weight is None:
			state_dim, state_feature_size = None, None
		else:
			state_feature_size, state_dim = state_weight.shape

		feature_net = _NatureCNN(
			in_channels=in_channels,
			n_flatten=n_flatten,
			feature_size=feature_size,
			state_dim=state_dim,
			state_feature_size=state_feature_size,
		)
		return _PPOExpert(feature_net=feature_net, hidden_size=hidden_size, n_act=n_act)

	def _validate_task_compatibility(self, state_dict) -> None:
		"""
		Refuse a task whose observation or action this expert was not trained on.

		Every check here is one that would otherwise pass silently or fail far from its
		cause. The camera and the state vector are the dangerous ones: a differently
		configured view, or `state_keys` in the wrong order, still yields a tensor of the
		right shape, and the expert then acts confidently on an observation that means
		something else.
		"""
		observation_dimension = self.task.observation_dimension
		if not isinstance(observation_dimension, dict):
			raise ValueError(
				"PPOPolicyModel needs a dict observation to pull an image and a state vector "
				f"out of, but {type(self.task).__name__} reports a single observation of "
				f"{tuple(observation_dimension)}."
			)

		if self.visual_key not in observation_dimension:
			raise ValueError(
				f"cfg.visual_key {self.visual_key!r} is not in the task's observation "
				f"({sorted(observation_dimension)})."
			)

		# (S, C, H, W): the frame stack is innermost in the batch dims, so the image itself is
		# the trailing three. `S` is deliberately unconstrained -- `policy` takes the newest
		# frame and the expert is memoryless.
		visual_dim = tuple(observation_dimension[self.visual_key])
		if len(visual_dim) != 4:
			raise ValueError(
				f"cfg.visual_key {self.visual_key!r} should be a stack of CHW frames, i.e. "
				f"(S, C, H, W), but the task reports {visual_dim}."
			)
		_, channels, height, width = visual_dim

		expected_channels = state_dict["feature_net.extractors.rgb.0.0.weight"].shape[1]
		if channels != expected_channels:
			raise ValueError(
				f"The task's {self.visual_key!r} has {channels} channels but this expert was "
				f"trained on {expected_channels}. The training script concatenates every "
				"camera it renders on the channel axis, so an expert trained with "
				"`--camera_view wrist` (one 3-channel wrist camera, since `wrist_only` drops "
				"the task's own) cannot be fed a two-camera observation or vice versa."
			)

		# The flattened width is the only thing that ties an expert to a resolution, so run
		# the conv stack rather than recomputing its arithmetic here.
		with torch.no_grad():
			# NCHW, not the NHWC `policy` passes: this is the bare conv stack, and the permute
			# that would consume NHWC lives one level up in `_NatureCNN.forward`.
			probe = torch.zeros(1, channels, height, width)
			n_flatten = self.agent.feature_net.extractors["rgb"][0](probe).shape[1]
		expected_n_flatten = state_dict["feature_net.extractors.rgb.1.weight"].shape[1]
		if n_flatten != expected_n_flatten:
			raise ValueError(
				f"The task renders {self.visual_key!r} at {height}x{width}, which flattens to "
				f"{n_flatten} features, but this expert's first linear layer takes "
				f"{expected_n_flatten}. Set the task's visual_observation_resolution to the "
				"`camera_resolution` in the expert's training_summary.json."
			)

		has_state_extractor = "feature_net.extractors.state.weight" in state_dict
		if has_state_extractor != (self.state_keys is not None):
			raise ValueError(
				f"cfg.state_keys is {self.state_keys!r} but this expert "
				f"{'has' if has_state_extractor else 'has no'} a state extractor. An expert "
				"trained with `--include_state` needs the keys its state vector was "
				"concatenated from; one trained with `--no-include_state` needs "
				"cfg.state_keys: null."
			)

		if self.state_keys is not None:
			missing = [key for key in self.state_keys if key not in observation_dimension]
			if missing:
				raise ValueError(
					f"cfg.state_keys {missing} are not in the task's observation "
					f"({sorted(observation_dimension)})."
				)
			state_dim = sum(observation_dimension[key][-1] for key in self.state_keys)
			expected_state_dim = state_dict["feature_net.extractors.state.weight"].shape[1]
			if state_dim != expected_state_dim:
				raise ValueError(
					f"cfg.state_keys {self.state_keys} give a {state_dim}-dim state vector, "
					f"but this expert's state extractor was trained on {expected_state_dim} "
					"dims. Under an image obs mode ManiSkill's flattened state is "
					"proprioception and the tcp pose only -- obs/agent/qpos, obs/agent/qvel "
					"and obs/extra/tcp_pose (9+9+7=25) for the PushCube experts -- and the "
					"order is that insertion order, not sorted. Either expose the same "
					"fields through the task's dataset_structure, or retrain the expert on "
					"the fields the task does expose."
				)

		action_dimension = tuple(self.task.action_dimension)
		expected_action_dim = state_dict["actor_mean.2.weight"].shape[0]
		if action_dimension != (expected_action_dim,):
			raise ValueError(
				f"The task's action is {action_dimension} but this expert emits "
				f"{(expected_action_dim,)}. The expert's control mode (and the task's "
				"frame_skip, which widens an action by concatenating primitive ones) has to "
				"match the `control_mode` in its training_summary.json."
			)

	# ------------------------------------------------------------------ policy model

	def _to_expert_obs(self, observation) -> typing.Tuple[dict, torch.Size]:
		"""
		Adapt one of this package's observations to the expert's `{rgb, state}` dict.

		Observations here are `(*batch_dims, S, *feature_dims)`; the expert takes a single
		leading batch dim and no time axis, so the mapping is "take the newest frame, collapse
		the batch dims". The batch dims come back with it so `policy` can restore them.
		"""
		try:
			keys = set(observation.keys())
		except AttributeError:
			raise ValueError(
				"PPOPolicyModel needs an image and (usually) a state vector, so `policy` must "
				f"be given a dict observation. Got {type(observation).__name__}."
			)

		if self.visual_key not in keys:
			raise ValueError(
				f"cfg.visual_key {self.visual_key!r} is not in this observation "
				f"({sorted(keys)})."
			)

		visual = observation[self.visual_key]              # (*batch, S, C, H, W)
		batch_dims = visual.shape[:-4]
		# `[..., -1, :, :, :]` is the newest frame: `make_env` stacks with the oldest first,
		# the same order `ManiSkillTrajectoryDataset` yields a window in.
		newest = visual[..., -1, :, :, :]                  # (*batch, C, H, W)
		# CHW -> NHWC, because `_NatureCNN.forward` permutes back itself. Left uint8: the
		# trained forward does the `.float() / 255`, and doing it here would double-scale.
		expert_obs = {
			"rgb": newest.reshape(-1, *newest.shape[-3:]).permute(0, 2, 3, 1)
		}

		if self.state_keys is not None:
			missing = [key for key in self.state_keys if key not in keys]
			if missing:
				raise ValueError(
					f"cfg.state_keys {missing} are not in this observation ({sorted(keys)})."
				)
			state = torch.cat(
				[observation[key][..., -1, :] for key in self.state_keys], dim=-1
			)                                              # (*batch, state_dim)
			expert_obs["state"] = state.reshape(-1, state.shape[-1]).float()

		return expert_obs, batch_dims

	def policy(self, o) -> torch.Tensor:
		expert_obs, batch_dims = self._to_expert_obs(o)
		action = self.agent.get_action(expert_obs)         # (prod(batch_dims), A)
		if self.cfg.clip_actions:
			action = action.clamp(self.action_low, self.action_high)
		return action.reshape(*batch_dims, action.shape[-1])

	def get_policy_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		# The single frame the expert actually reads, in this package's CHW layout rather than
		# the NHWC the wrapped forward wants -- `_PPOPolicyForFLOPS` does that permute, so the
		# shape reported here is the one a caller would slice out of an observation.
		_, channels, height, width = tuple(self.task.observation_dimension[self.visual_key])
		state_dim = (
			None if self.state_keys is None
			else sum(self.task.observation_dimension[key][-1] for key in self.state_keys)
		)
		return (
			(channels, height, width),
			_PPOPolicyForFLOPS(self.agent, state_dim),
		)

	def requires_grad_(self, requires_grad):
		return super().requires_grad_(requires_grad and not self.cfg.freeze)

	def save_to_file(self, filepath:str) -> None:
		# Frozen: these weights are still exactly the checkpoint this was built from, so
		# point at it instead of copying it. See s2p.lib.checkpointing.
		if self.cfg.freeze:
			save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
			return

		# The expert's own state_dict, not this wrapper's, so what is written stays in the
		# format `ppo_visual_expert_fast.py` reads and `load_from_file` below round-trips.
		torch.save(self.agent.state_dict(), filepath)

	def load_from_file(self, filepath:str) -> None:
		# May be a reference written by `save_to_file` above rather than weights.
		filepath = resolve_checkpoint_path(filepath)
		self.agent.load_state_dict(torch.load(filepath, map_location=self.cfg.device))


class _PPOPolicyForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats the expert's action computation for one observation.
	Used for counting the FLOPs of the policy path.

	Takes the frame in this package's CHW layout and a state vector, so that its input is
	what `get_policy_function` reports rather than the expert's internal NHWC convention.
	"""

	def __init__(self, agent, state_dim):
		super().__init__()
		self.agent = agent
		self.state_dim = state_dim

	def forward(self, visual, state=None):
		expert_obs = {"rgb": visual.permute(0, 2, 3, 1)}
		if self.state_dim is not None:
			if state is None:
				raise ValueError(
					f"This expert reads a {self.state_dim}-dim state vector alongside the "
					"image, so `state` is required."
				)
			expert_obs["state"] = state
		return self.agent.get_action(expert_obs)
