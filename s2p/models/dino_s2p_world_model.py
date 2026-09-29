import hashlib
import os
import typing

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path


# The observation key `DINOV3PassthroughEncoderModel.encode` short-circuits on. Handing S2P's
# encoder a dict carrying only this key is what makes it skip its own DINOv3 backbone and take
# the DINO-WM visual tokens as the features it would otherwise have computed.
_S2P_FEATURE_KEY = "dino_patch_features"

# Written into every manifest `save_to_file` writes, so `load_from_file` can tell one apart.
_MANIFEST_MAGIC = "s2p.DINOS2PWorldModel.manifest.v1"
# Manifests saved before the project was renamed from TSD carry this magic and name their
# components `tsd_encoder` / `tsd_reward` / `tsd_value`; `load_from_file` still reads them.
_LEGACY_MANIFEST_MAGICS = ("tsd.DINOTSDWorldModel.manifest.v1",)


class DINOS2PWorldModel(
	EncoderModelBase,
	DynamicsModelBase,
	RewardModelBase,
	StateValueModelBase,
	torch.nn.Module
):
	"""
	DINO-WM's latent and dynamics, scored by S2P's reward and value.

	* `encode` / `dynamics` are DINO-WM's own (`DINOWorldModel`), unchanged: the state is the
	  flattened DINO-WM window, and MPPI rolls it forward with DINO-WM's predictor.
	* `reward` / `state_value` pull the visual patch tokens back out of that window, hand them
	  to S2P's encoder in place of the DINOv3 features it would have computed from pixels, and
	  return whatever S2P's reward (with the planner's action) and value say over the
	  resulting S2P latent.

	The two meet in DINOv3 feature space: DINO-WM's visual tokens are dinov3_vits16
	`x_norm_patchtokens`, the same features S2P's `DINOV3PassthroughEncoderModel` produces
	(see `dino_wm.models.dinov3.DinoV3Encoder`), and the predictor is trained to predict them.

	### S2P's encoder has to be vision-only

	Only the visual component of the window is handed to S2P. DINO-WM's proprio is a 10-dim
	learned embedding of the normalized 25-dim proprio, which no S2P encoder was trained on and
	which cannot be inverted back to the raw vector. So the S2P encoder here must be one whose
	teacher reads DINO features alone, i.e. `teacher_encoder` is the passthrough encoder
	itself rather than `vision_proprioception_joint_encoder`. `__init__` probes this and fails
	if the teacher asks for any other observation key.

	### Everything is frozen

	Nothing here trains. Each component loads its own weights from its own config and keeps
	its own `freeze` behaviour; this class only routes between them.
	"""

	def __init__(
			self,
			cfg,
			task:TaskBase,
			world_model:typing.Union[EncoderModelBase, DynamicsModelBase],
			s2p_encoder:EncoderModelBase,
			s2p_reward:RewardModelBase,
			s2p_value:StateValueModelBase,
	):
		super().__init__(cfg=cfg)
		self.cfg = cfg
		self.task = task

		self.world_model = world_model
		self.s2p_encoder = s2p_encoder
		self.s2p_reward = s2p_reward
		self.s2p_value = s2p_value

		self.latent_dim = self.world_model.latent_dim
		self.action_dim = self.task.action_dimension[-1]

		# Probe the whole reward/value path once, so a S2P encoder that wants more than the
		# visual tokens fails here with an explanation instead of as a KeyError mid-plan.
		probe_state = torch.zeros(1, self.latent_dim, device=self.cfg.device)
		try:
			with torch.no_grad():
				probe_s2p_latent = self._s2p_latent(probe_state)
		except KeyError as error:
			raise ValueError(
				f"S2P's encoder asked for observation key {error.args[0]!r}, but DINOS2PWorldModel "
				f"can only give it DINO-WM's visual tokens (as {_S2P_FEATURE_KEY!r}). Use a S2P "
				"checkpoint trained with a vision-only teacher -- `teacher_encoder` set to "
				"`frozen_dinov3_vits_passthrough_patch_encoder` -- rather than one whose teacher "
				"also reads proprioception."
			) from error
		self.s2p_latent_dim = probe_s2p_latent.shape[-1]
		with torch.no_grad():
			self.s2p_reward.reward(probe_s2p_latent, torch.zeros(1, self.action_dim, device=self.cfg.device))
			self.s2p_value.state_value(probe_s2p_latent)

	# ---------------------------------------------------------------- DINO-WM -> S2P

	def _visual_tokens(self, s:torch.Tensor) -> torch.Tensor:
		"""(*batch_dims, latent_dim) -> (prod(batch_dims), num_hist, patches, visual_dim)"""
		z = self.world_model._unflatten_latent(s)
		# DINO-WM's own split, so both `concat_dim` layouts come from the code that defines them.
		z_obs, _ = self.world_model.dino_wm.separate_emb(z)
		return z_obs["visual"]

	def _s2p_latent(self, s:torch.Tensor) -> torch.Tensor:
		"""(*batch_dims, latent_dim) DINO-WM state -> (*batch_dims, s2p_latent_dim) S2P latent"""
		batch_dims = s.shape[:-1]
		visual = self._visual_tokens(s)
		s2p_latent = self.s2p_encoder.encode({_S2P_FEATURE_KEY: visual})
		return s2p_latent.reshape(*batch_dims, s2p_latent.shape[-1])

	# ------------------------------------------------------------ encoder / dynamics

	def encode(self, observation, previous_state:torch.Tensor=None, action:torch.Tensor=None) -> torch.Tensor:
		return self.world_model.encode(observation, previous_state=previous_state, action=action)

	def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		return self.world_model.get_encoding_function()

	def dynamics(self, s:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
		return self.world_model.dynamics(s, a)

	def get_dynamics_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		return self.world_model.get_dynamics_function()

	# ---------------------------------------------------------------- reward / value

	def reward(self, s:torch.Tensor, a:torch.Tensor) -> torch.Tensor:
		# The action goes to S2P as the task action, untouched: DINO-WM's normalization and
		# frameskip tiling belong to its own predictor, not to S2P's reward.
		return self.s2p_reward.reward(self._s2p_latent(s), a)

	def state_value(self, s:torch.Tensor) -> torch.Tensor:
		return self.s2p_value.state_value(self._s2p_latent(s))

	def get_reward_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		reward_input_dims, reward_head = self.s2p_reward.get_reward_function()
		module = _DINOS2PHeadForFLOPS(
			world_model=self,
			student=self._student_module(),
			head=reward_head,
			head_input_dim=reward_input_dims[-1],
			s2p_latent_dim=self.s2p_latent_dim,
			latent_dim=self.latent_dim,
		)
		return ((self.latent_dim + self.action_dim,), module)

	def get_state_value_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		value_input_dims, value_head = self.s2p_value.get_state_value_function()
		module = _DINOS2PHeadForFLOPS(
			world_model=self,
			student=self._student_module(),
			head=value_head,
			head_input_dim=value_input_dims[-1],
			s2p_latent_dim=self.s2p_latent_dim,
			latent_dim=self.latent_dim,
		)
		return ((self.latent_dim,), module)

	def _student_module(self) -> torch.nn.Module:
		# Only the student runs on this path: the vision-only teacher is the passthrough
		# encoder, which returns the visual tokens it is handed without touching its backbone.
		# `TeacherStudentEncoderModel.get_encoding_function` would wrap that backbone in too.
		_, student = self.s2p_encoder.student_encoder.get_encoding_function()
		return student

	# ------------------------------------------------------------------------ extras

	def decode(self, s:torch.Tensor) -> torch.Tensor:
		return self.world_model.decode(s)

	def requires_grad_(self, requires_grad):
		# Each component applies its own `freeze` on top of this one's.
		requires_grad = requires_grad and not self.cfg.freeze
		for component in self._components().values():
			component.requires_grad_(requires_grad)
		return self

	def _components(self) -> typing.Dict[str, torch.nn.Module]:
		return {
			"world_model": self.world_model,
			"s2p_encoder": self.s2p_encoder,
			"s2p_reward": self.s2p_reward,
			"s2p_value": self.s2p_value,
		}

	def save_to_file(self, filepath:str) -> None:
		"""
		Write each component to its own file beside `filepath`, and a manifest at `filepath`.

		`AgentModel.save_to_folder` gives this object one slot per role it fills, but it is
		four models with four checkpoints, and each already knows how to save itself -- a
		frozen one as a small reference to the file it was loaded from. So each writes its own
		entry (`reward.s2p_value.pt`, ...) and the slot itself holds only the manifest naming
		them.
		"""
		stem, extension = os.path.splitext(filepath)
		files = {}
		for name, component in self._components().items():
			component_path = f"{stem}.{name}{extension}"
			component.save_to_file(component_path)
			files[name] = os.path.basename(component_path)
		torch.save({"magic": _MANIFEST_MAGIC, "files": files}, filepath)

	def load_from_file(self, filepath:str) -> None:
		filepath = resolve_checkpoint_path(filepath)
		manifest = torch.load(filepath, map_location="cpu", weights_only=False)
		if not (isinstance(manifest, dict) and manifest.get("magic") in (_MANIFEST_MAGIC, *_LEGACY_MANIFEST_MAGICS)):
			raise ValueError(
				f"{filepath!r} is not a DINOS2PWorldModel manifest. This model is saved as one "
				"file per component plus a manifest naming them; see `save_to_file`."
			)
		folder = os.path.dirname(filepath)
		components = self._components()
		for name, filename in manifest["files"].items():
			if name.startswith("tsd_"):
				name = "s2p_" + name[len("tsd_"):]
			components[name].load_from_file(os.path.join(folder, filename))


class _DINOS2PHeadForFLOPS(torch.nn.Module):
	"""
	Module whose forward pass repeats `DINOS2PWorldModel.reward` / `.state_value` from a flat
	DINO-WM input: visual tokens out of the window, S2P's student over them, then the S2P
	head. Used for counting the FLOPs of the reward or value path only.

	`head` is whatever the S2P model reports as its own FLOPs module, fed the way S2P's own
	`FLOPSEvaluator` feeds it: its reported input width, with the S2P latent first and the
	rest (the action, or TD-MPC2's action and task channels) zero-padded. Shapes, not values,
	decide the FLOPs, and this keeps the head's count identical to what a S2P run reports.

	The composite model is held *outside* the module tree (`object.__setattr__`) and used only
	for its `_visual_tokens` slicing, so the DINO-WM predictor, which this path never runs, is
	not counted into the parameters. Same reasoning as `_DINOWMRewardHeadForFLOPS`.
	"""

	def __init__(self, world_model, student, head, head_input_dim, s2p_latent_dim, latent_dim):
		super().__init__()
		object.__setattr__(self, "world_model", world_model)
		self.student = student
		self.head = head
		self.head_input_dim = head_input_dim
		self.s2p_latent_dim = s2p_latent_dim
		self.latent_dim = latent_dim

	def forward(self, flat_input:torch.Tensor):
		latent = flat_input[..., :self.latent_dim]
		visual = self.world_model._visual_tokens(latent)            # (b, num_hist, patches, dim)
		tokens = visual.reshape(visual.shape[0], -1, visual.shape[-1])
		s2p_latent = self.student(tokens)                           # (b, s2p_latent_dim)
		padding = torch.zeros(
			s2p_latent.shape[0], self.head_input_dim - self.s2p_latent_dim, device=s2p_latent.device
		)
		return self.head(torch.cat([s2p_latent, padding], dim=-1))


_registry: dict = {}

def get_or_create(cfg, task, world_model, s2p_encoder, s2p_reward, s2p_value) -> "DINOS2PWorldModel":
	"""
	One instance per config, however many agent roles it is wired into.

	An evaluation config hands this one object to the encoder, dynamics, reward and value
	slots, and Hydra instantiates each of those references separately. The config node
	therefore carries `_recursive_: false`, so everything below arrives as a config and is
	only built on a cache miss -- otherwise each slot would build its own S2P encoder, two
	DINOv3 backbones apiece, before the cache was ever consulted. DINO-WM itself memoizes in
	its own `get_or_create`, so rebuilding it here returns the shared instance.
	"""
	parts = (cfg, task, world_model, s2p_encoder, s2p_reward, s2p_value)
	key = hashlib.md5("".join(OmegaConf.to_yaml(part) for part in parts).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = DINOS2PWorldModel(
			cfg=cfg,
			task=instantiate(task) if not isinstance(task, TaskBase) else task,
			world_model=instantiate(world_model),
			s2p_encoder=instantiate(s2p_encoder),
			s2p_reward=instantiate(s2p_reward),
			s2p_value=instantiate(s2p_value),
		)
	return _registry[key]
