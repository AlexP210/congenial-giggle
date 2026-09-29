import hashlib
import math

from omegaconf import OmegaConf
import torch
import numpy as np
import gymnasium as gym
from tensordict import TensorDict

from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.lib.maniskill_transition_dataset import (
	ManiSkillTrajectoryDataset,
	_get_nested,
	stage_to_slurm_tmpdir,
)
from s2p.lib.seeding import seed_spaces

import mani_skill.envs
from mani_skill.utils.registration import REGISTERED_ENVS

# The env itself -- camera views, the wrist Panda, device pinning, termination handling and frame
# stacking -- is defined once in environments/custom_maniskill_tasks, and shared with the dataset
# recorder (tools/ppo_stages_fast.py) and the converter (tools/replay_trajectory.py), so a policy
# cannot be trained against a differently-configured env than its data was recorded through. What
# stays here is what only S2P wants: the TensorDict view of the observation dict below.
from custom_maniskill_tasks import make_env, resolve_sim_backend


class AttributePassthroughSubset(torch.utils.data.Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)


class ManiSkillTask(OnlineTaskBase, OfflineTaskBase):

	# Shared across all ManiSkillTask instances so that identical dataset configs
	# (e.g. across separate online/offline task instantiations) load the underlying
	# h5 trajectory data only once.
	_dataset_registry: dict = {}

	def __init__(self, cfg):
		super().__init__(cfg)
		self.cfg = cfg

		# One knob for both halves: the env acts in chunks of `frame_skip` primitive actions and
		# the dataset serves windows on the same stride, so the assert below (env action dim vs
		# dataset action dim) fails loudly if they ever disagree.
		self.frame_skip = cfg.frame_skip

		# The env is optional when there is a recording to fall back on: MIG partitions (Nibi's
		# h100_*g.*gb slices) expose no Vulkan device, so sapien cannot build a renderer there at
		# any sim backend, and offline training needs nothing from the env that the recording
		# cannot supply. `action_limits` is then left unset, so anything that does need the env
		# (a planner clamping proposals, an online rollout) still fails, just later and by name.
		self.env = None
		self._observation_dimension = {}
		try:
			self.env = self.make_env()
			# Per-env, i.e. with the batch axis dropped: these describe one env's action and
			# observation, and must not move when a component asks for a different `num_envs`.
			# See `OnlineTaskBase._per_env_dims_from_space`.
			self._action_dimension = self._per_env_dims_from_space(self.env.action_space)
			# ManiSkill's controller clips to this box itself (`BaseController.set_action`), so it
			# is exactly what an executed action is held to.
			self._action_limits = self._per_env_limits_from_space(self.env.action_space)
			self._observation_dimension = self._per_env_dims_from_space(self.env.observation_space)
		except Exception as error:
			# An online-only task has nothing to fall back on, so the error is the real failure.
			if self.cfg.data_path is None:
				raise
			# Named, not swallowed silently: an online-capable node failing here is a real bug,
			# and is otherwise indistinguishable from a MIG partition that cannot render.
			print(f"[ManiSkillTask] Could not load env: {type(error).__name__}: {error}")

		if self.cfg.data_path is not None:
			obs_dims_from_dataset, action_dims_from_dataset = self._dims_from_dataset(self.training_dataset)
			self._observation_dimension.update(obs_dims_from_dataset)
			if self.env is None:
				self._action_dimension = action_dims_from_dataset
			else:
				assert self._action_dimension == action_dims_from_dataset
		# Read off the task's own registered time limit rather than assuming 50: that is right for
		# PushCube/PlaceSphere/LiftPegUpright but not for e.g. PegInsertionSide-v1, which is 100.
		# In macro steps, since that is what one env interaction is: the last chunk is counted even
		# when frame_skip does not divide the limit, because the episode does reach it.
		primitive_steps = REGISTERED_ENVS[self.cfg.task_name].max_episode_steps
		self._episode_length = math.ceil(primitive_steps / self.frame_skip)
		self._task_name = self.cfg.task_name

	def make_env(self, num_envs: int = None, max_episode_steps: int = None):
		"""`num_envs` copies of the task, built by `custom_maniskill_tasks.make_env`.

		`num_envs=None` means `cfg.num_envs`, the task's own default; an online component that
		wants a different width passes its own, so the trainer can collect on sixteen envs while
		an evaluator scores on eight.

		`max_episode_steps=None` keeps the task's registered time limit (50 for PushCube,
		PlaceSphere and LiftPegUpright; 100 for PegInsertionSide-v1). A number replaces it, in
		ManiSkill's own unit -- *primitive* steps, the same unit `REGISTERED_ENVS[...]
		.max_episode_steps` is in -- so the env truncates after `max_episode_steps / frame_skip`
		of the macro steps a caller actually takes, and the number keeps its meaning when
		`frame_skip` changes. `self._episode_length` is not touched: it describes the task, and
		this env is one component's view of it.

		`ManiSkillWrapper` goes in as an observation adapter, i.e. between the env and the frame
		stack, so it sees single frames and the stack is built out of its TensorDicts. It keeps
		ManiSkill's leading num_envs axis, so the stack goes on at `frame_axis=1` and each leaf
		comes out as (num_envs, num_frames, *feature_dims) -- i.e. per env, exactly the
		(num_frames, *feature_dims) layout `ManiSkillTrajectoryDataset` yields for a training
		sample.
		"""
		num_envs = self.cfg.num_envs if num_envs is None else int(num_envs)
		self._check_backend_supports(num_envs)
		observations_to_expose = OmegaConf.to_container(self.cfg.dataset_structure)["obs"]
		# Forwarded into `gym.make` only when asked for: passing `max_episode_steps=None`
		# through would override the task id's registered limit with None rather than defer
		# to it, which is the same trap `control_mode=None` carries in `make_env`.
		env_kwargs = {} if max_episode_steps is None else {"max_episode_steps": int(max_episode_steps)}
		# Only LiftPegUpright-v1.1 takes `distractors`, so it is forwarded only when asked for:
		# every other task (and every existing config, which has no such key) builds as before.
		if self.cfg.get("distractors", False):
			env_kwargs["distractors"] = True
		return make_env(
			self.cfg.task_name,
			obs_mode=self.cfg.observation_mode,
			control_mode=self.cfg.control_mode,
			num_envs=num_envs,
			camera_view=self.cfg.camera_view,
			lighting=self.cfg.lighting,
			camera_resolution=self.cfg.visual_observation_resolution,
			n_frames=self.cfg.num_frames,
			frame_skip=self.frame_skip,
			frame_axis=1,
			obs_wrappers=[
				lambda env: ManiSkillWrapper(env, observations_to_expose, num_envs=num_envs),
			],
			render_mode="rgb_array" if "rgb" in self.cfg.observation_mode else None,
			sim_backend=self.cfg.sim_backend,
			parallel_in_single_scene=False,
			**env_kwargs,
		)

	def _check_backend_supports(self, num_envs: int) -> None:
		"""Refuse a `num_envs` this task's `cfg.sim_backend` cannot serve, and say why.

		Two reasons, and the first is the one that bites: sapien enables PhysX once per process
		("GPU PhysX can only be enabled once before any other code involving PhysX"), so every env
		built here -- this task's own, the trainer's, each evaluator's -- has to agree on a
		backend. Leaving `sim_backend: null` does not agree with itself: ManiSkill's "auto" is
		physx_cpu at one env and physx_cuda above it, so a single-env probe env followed by a
		parallel trainer env is exactly the combination that dies. The second is that physx_cpu
		steps one scene and cannot run copies at all.

		So parallel collection is a decision about the whole run, made once in the task config,
		rather than something `trainer.cfg.num_envs` can turn on by itself.
		"""
		# The `null` default resolves *per env count*, so the check has to compare this
		# request against the env the task already built for itself rather than just look at
		# what this one env count resolves to on its own. `sim_backend: null` with
		# cfg.num_envs=1 and a `num_envs > 1` component passes the physx_cuda test below
		# while still disagreeing with the physx_cpu env already standing -- and sapien then
		# fails with "GPU PhysX can only be enabled once before any other code involving
		# PhysX", which names neither of the two configs that actually disagreed.
		own_backend = resolve_sim_backend(self.cfg.sim_backend, self.cfg.num_envs)
		backend = resolve_sim_backend(self.cfg.sim_backend, num_envs)
		if backend != own_backend:
			raise ValueError(
				f"num_envs={num_envs} resolves `task.cfg.sim_backend={self.cfg.sim_backend!r}` "
				f"to {backend}, but this task's own env was built at "
				f"cfg.num_envs={self.cfg.num_envs}, which resolves to {own_backend}. Every env "
				f"in the process shares one PhysX backend, so it cannot be left to ManiSkill's "
				f"per-env-count default: set it explicitly, e.g. "
				f"`task.cfg.sim_backend=physx_{self.cfg.batch_device}`."
			)

		if num_envs <= 1:
			return
		if not backend.startswith("physx_cuda"):
			raise ValueError(
				f"num_envs={num_envs} needs the GPU sim, but `task.cfg.sim_backend` is "
				f"{self.cfg.sim_backend!r} (which resolves to {backend}). Set it explicitly, e.g. "
				f"`task.cfg.sim_backend=physx_{self.cfg.batch_device}`: every env in the process "
				"shares one PhysX backend, and this task builds its own at cfg.num_envs, so the "
				"backend cannot be left to ManiSkill's per-env-count default."
			)

	def get_control_interval(self):
		# ManiSkill's control_freq is 20 Hz, and one env step now applies `frame_skip` of those
		return 0.05 * self.frame_skip

	def sample_goal_observation(self):
		"""
		The terminal observation of a randomly chosen successful demonstration, for
		goal-conditioned world models (DINO-WM scores states by latent distance to a goal
		rather than with a reward head).

		The success flag comes from the trajectory json's per-episode `success`, which is
		ManiSkill's success predicate evaluated on the last step — verified equal to
		`traj_*/success[-1]` for every episode in the PushCube recording. That matters
		because the predicate does not latch (see `ManiSkillWrapper.step`): an episode can
		pass through the goal and leave it again, so "was ever successful" is not the same
		as "ends in a goal state", and only the latter makes a usable goal.

		Returned in the same layout as an observation from `make_env`: a TensorDict whose
		leaves are (num_frames, *feature_dims), stacked back from the end of the episode.

		Episodes are drawn from the whole recording, not just the training split — that
		split is over horizon windows, not episodes, so it does not partition them anyway.

		None when the task was configured without a recording (`cfg.data_path: null`, which
		is what every evaluation config here sets so that the dataset's stored dimensions do
		not override the env's). There is no recording to draw a terminal observation from,
		and that is the same answer `TaskBase.sample_goal_observation` gives for a task that
		defines no goal at all — a goal-conditioned model then stays unset and says so when
		something asks it to score a state, rather than this raising a `TypeError` from
		inside the dataset loader at construction time.
		"""
		if self.cfg.data_path is None:
			return None

		dataset = self.training_dataset.dataset
		successful = [
			index for index, episode in enumerate(dataset.episode_metadata)
			if episode.get("success")
		]
		if not successful:
			raise ValueError(
				f"No episode in {self.cfg.data_path} ends in success, so there is no goal "
				"observation to sample. Either the recording holds only failures, or its "
				"json carries no per-episode `success` flag."
			)
		return dataset.stacked_observation(int(np.random.choice(successful)), step=-1)

	def make_dataset(self):
		key = hashlib.md5(OmegaConf.to_yaml(self.cfg).encode()).hexdigest()
		if key in self._dataset_registry:
			return self._dataset_registry[key]

		data_path = self.cfg.data_path
		json_path = self.cfg.json_path
		if self.cfg.local_dir is not None and self.cfg.copy_dataset_to_local_dir:
			import shutil
			import os
			print(f"Copying {self.cfg.data_path} to {self.cfg.local_dir}")
			shutil.copy(self.cfg.data_path, self.cfg.local_dir)
			print(f"Copying {self.cfg.json_path} to {self.cfg.local_dir}")
			shutil.copy(self.cfg.json_path, self.cfg.local_dir)
			data_path = os.path.join(self.cfg.local_dir, os.path.basename(self.cfg.data_path))
			json_path = os.path.join(self.cfg.local_dir, os.path.basename(self.cfg.json_path))
		elif data_path is not None:
			# The h5 is the only file worth staging: in mmap mode (`load=False`) every sample
			# faults its pages straight off whichever filesystem it sits on, so leaving it on
			# home/project/scratch puts the network in the middle of every batch -- which is
			# exactly what `local_dir: null` used to do, at ~0.2 CPU cores and ~0.5 it/s. The
			# json is a few MB read once at load time, so it stays where it is.
			data_path = str(stage_to_slurm_tmpdir(data_path))

		dataset = ManiSkillTrajectoryDataset(
			dataset_file=data_path,
			json_file=json_path,
			load_count=-1,
			success_only=False,
			load_device=self.cfg.load_device,
			batch_device=self.cfg.batch_device,
			horizon=self.cfg.horizon,
			num_frames=self.cfg.num_frames,
			frame_skip=self.frame_skip,
			structure=OmegaConf.to_container(self.cfg.dataset_structure),
			load=self.cfg.load
		)
		train_size = int(0.8 * len(dataset))
		val_size = len(dataset) - train_size
		training_subset, validation_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
		training_dataset = AttributePassthroughSubset(dataset, training_subset.indices)
		validation_dataset = AttributePassthroughSubset(dataset, validation_subset.indices)
		self._dataset_registry[key] = (training_dataset, validation_dataset)
		return self._dataset_registry[key]

_registry: dict = {}

def get_or_create(cfg) -> "ManiSkillTask":
	key = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = ManiSkillTask(cfg=cfg)
	return _registry[key]

class ManiSkillWrapper(gym.Wrapper):
	"""
	Exposes the observation paths named in `observations_to_expose` as top-level keys
	of a TensorDict, dropping the rest of ManiSkill's raw (nested, batched-by-num_envs)
	observation structure. Images are transposed from (num_envs, H, W, C) to
	(num_envs, C, H, W); everything else keeps the layout ManiSkill gave it.

	ManiSkill's leading `num_envs` axis is kept, at `num_envs=1` as much as at 16 — this is
	the batched env contract `OnlineTaskBase.make_env` documents, and it is why `make_env`
	stacks frames at `frame_axis=1`. It used to be squeezed away here, which is what forced
	every caller to `unsqueeze` an env axis back on; keeping it means one observation layout
	for every task family and no `num_envs == 1` special case anywhere downstream. The
	squeeze could not simply be made conditional: `(E,H,W,C).squeeze(0).permute(2,0,1)` does
	not raise on a batched observation, it silently returns garbage.

	Paths are relative to a ManiSkill trajectory group, same as `structure` in
	ManiSkillTrajectoryDataset — "obs/..." is looked up in the live env's observation
	dict, while "env_states/..." (e.g. privileged actor/articulation poses recorded by
	RecordEpisode) has no live-env observation counterpart and is instead looked up in
	env.unwrapped.get_state_dict(), which uses the same nesting and actor/articulation
	names as the recorded trajectories.

	Termination is suppressed underneath this wrapper rather than in it, by
	`custom_maniskill_tasks.IgnoreTerminations` (and, for the `-v1.1` task ids, by the task
	itself), so an episode is never cut short by a momentary success.
	"""
	def __init__(self, env: gym.Env, observations_to_expose, num_envs: int = 1):
		super().__init__(env)
		self._obs_to_expose = observations_to_expose
		self.num_envs = num_envs

		# ManiSkill batches its *action* space only above one env (`BaseController.__init__`),
		# so at num_envs=1 it is `(A,)` while the observations coming out of here are `(1, ...)`.
		# Batch it explicitly so the two agree and `step` is always handed `(num_envs, A)` --
		# `BaseEnv.step` accepts that at one env as readily as at many. `single_action_space` is
		# read through rather than recomputed because `FrameSkip`, below this wrapper, has
		# already widened it by `frame_skip`.
		self.single_action_space = env.get_wrapper_attr("single_action_space")
		self.action_space = gym.vector.utils.batch_space(self.single_action_space, n=num_envs)
		# `rand_act` below samples these, and a gymnasium Space seeds its own RNG from OS entropy
		# rather than from the global numpy one -- so without this the seed-step rollouts that
		# fill the replay buffer differ every run. See s2p/lib/seeding.py::seed_spaces.
		seed_spaces(self.action_space, self.single_action_space)

		state_dict = self.env.unwrapped.get_state_dict()
		new_spaces = {}
		for nickname, path in self._obs_to_expose.items():
			if path.startswith("env_states/"):
				try:
					value = _get_nested(state_dict, path.removeprefix("env_states/"))
				except (KeyError, TypeError):
					continue
				arr = value.cpu().numpy()
				new_spaces[nickname] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=arr.dtype)
				continue

			space = env.observation_space
			try:
				for key in (path.removeprefix("obs/") if path.startswith("obs/") else path).split("/"):
					space = space[key]
			except:
				continue
			if space.low.ndim == 4:
				# (num_envs, H, W, C) → (num_envs, C, H, W)
				new_low = space.low.transpose(0, 3, 1, 2)
				new_high = space.high.transpose(0, 3, 1, 2)
				new_spaces[nickname] = gym.spaces.Box(low=new_low, high=new_high, dtype=space.dtype)
			else:
				new_spaces[nickname] = space
		self.observation_space = gym.spaces.Dict(new_spaces)

		# ManiSkill's camera sensors (rgb/depth/segmentation) declare a uint8 dtype on
		# their observation_space leaf; other obs (feature maps, poses) are float. Use
		# that as the generic signal for "this nickname is an image" rather than shape,
		# since e.g. dino_patch_features is also a (num_envs, H, W, C) tensor but isn't an image.
		self._image_nicknames = {
			nickname for nickname, space in new_spaces.items()
			if space.dtype == np.uint8
		}

	def _resolve(self, path, obs, state_dict):
		if path.startswith("env_states/"):
			return _get_nested(state_dict, path.removeprefix("env_states/"))
		return _get_nested(obs, path.removeprefix("obs/") if path.startswith("obs/") else path)

	def _transform_obs(self, obs):
		state_dict = self.env.unwrapped.get_state_dict()
		result = {}
		for nickname, path in self._obs_to_expose.items():
			try:
				x = self._resolve(path, obs, state_dict)
				if isinstance(x, torch.Tensor) and x.ndim == 4:
					x = x.permute(0, 3, 1, 2)  # (num_envs, H, W, C) → (num_envs, C, H, W)
				if nickname in self._image_nicknames and isinstance(x, torch.Tensor):
					x = x.to(torch.uint8)
				result[nickname] = x
			except:
				continue
		# An explicit batch size, so that a caller can slice this observation by env
		# (`observation[i]`, or a stacked window's `[:, i]`) rather than having to reach into
		# every leaf. Inferring it would give `[]`, since the leaves share no longer prefix.
		return TensorDict(result, batch_size=[self.num_envs])

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		return self._transform_obs(obs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return self._transform_obs(obs), reward, terminated, truncated, info

	def render(self):
		# The human render camera is per-env too; a video is of one episode, so report env 0's.
		img = np.array(self.env.render()[0].cpu())
		return img

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def __getattr__(self, name):
		"""
		If this env does not have the attribute, then we try to
		recursively access that attribute from inner envs.

		gymnasium >= 1.0 dropped `Wrapper.__getattr__`, so without this a caller reaching
		through for ManiSkill's own API (`get_state_dict`, `control_freq`, ...) sees only
		the outermost wrapper.
		"""
		if name.startswith("_"):
			raise AttributeError(name)
		env = self.env
		while not hasattr(env, name):
			if hasattr(env, 'env'): # while the env is still wrapped,
				env = env.env
			else: # reached the innermost env and still didn't find it.
				raise AttributeError(f'{env} has no attribute {name}.')
		return getattr(env, name) # reached if env **has** attribute name.
