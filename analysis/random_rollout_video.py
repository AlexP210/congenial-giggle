"""Record videos of a random-action trajectory in a task's environment.

Launched with a hydra config exactly like `analysis.py`, but it builds nothing except the
task: no model, no checkpoint, no evaluators. It makes the env, steps it with uniformly
random actions, and writes two mp4s of the same rollout:

	<stem>.observation.mp4   what the policy sees -- the task's sensor camera at its
	                         configured pose and resolution
	<stem>.render.mp4        the human render camera, i.e. whatever `env.render()` draws,
	                         usually a wider and higher-resolution view of the scene

A quick way to check both what a task config actually feeds the model and what the scene it
came from looked like.

	python random_rollout_video.py --config-name=evaluate_visual_online_stochastic \
		data_dir=/path/to/datasets checkpoint_dir=/tmp output_dir=/tmp

The usual mandatory root-config keys still have to be supplied, since the task resolves them
while instantiating -- notably `data_dir`, because the tasks read their observation and
action dimensions off the offline dataset.

Script-specific overrides (all optional; note the leading `+`, as these are not part of the
root configs):

	+num_episodes=3       how many random episodes to record (default 1)
	+max_steps=100        cap on steps per episode (default: the task's episode length)
	+video_fps=15         playback rate of the written mp4s (default 15, as in the wandb logs)
	+video_path=foo.mp4   stem for both files, whose extension is replaced by
	                      `.observation.mp4` / `.render.mp4` (default random_rollout_<task>)
"""

import os

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from s2p import PROJECT_ROOT
from s2p.tasks.base.online_task_base import OnlineTaskBase


def latest_frame(image: torch.Tensor) -> np.ndarray:
	"""Reduce one stacked-frame observation entry to its newest frame, as (H, W, C) uint8.

	A video is of one episode, so where the entry carries an env axis (every online env is
	batched -- see `OnlineTaskBase.make_env`) this follows env 0, the same one `env.render()`
	reports.
	"""
	if image.ndim == 5:
		# (num_envs, num_frames, C, H, W)
		image = image[0]
	if image.ndim == 4:
		# (num_frames, C, H, W), as produced by the FrameStack wrapper
		image = image[-1]
	elif image.ndim == 3 and image.shape[0] not in (1, 3):
		# (num_frames * C, H, W) -- the Pixels wrapper concatenates its stack along channels
		image = image[-3:]
	return image.detach().cpu().numpy().transpose(1, 2, 0)


def observation_frame(env, obs) -> np.ndarray:
	"""Return the current observation image as an (H, W, 3) uint8 array.

	This is what the policy actually sees -- the task's sensor camera at its configured pose
	and resolution -- rather than the human render camera that `env.render()` draws from.

	Both observation layouts the tasks produce are handled: a TensorDict of named entries
	(ManiSkill), of which the uint8 ones are the camera images, and a bare stacked-pixel
	tensor (visual dm_control). Frame stacking is undone by keeping only the newest frame,
	and multiple cameras are tiled left to right.
	"""
	if hasattr(obs, "keys"):
		images = [obs[key] for key in sorted(obs.keys()) if obs[key].dtype == torch.uint8]
	else:
		images = [obs] if obs.dtype == torch.uint8 else []

	if len(images) == 0:
		raise RuntimeError(
			"this task's observations contain no camera image -- use a task config with an "
			"image observation mode (e.g. task=maniskill_push_cube or task=dm_control_visual)"
		)

	return np.concatenate([latest_frame(image) for image in images], axis=1)


def render_frame(env, obs) -> np.ndarray:
	"""Return the human render camera's view as an (H, W, 3) uint8 array.

	Unlike the observation, this is the free viewing camera the task renders for people, so it
	exists even for tasks whose observations are plain state vectors.
	"""
	frame = env.render()
	if isinstance(frame, torch.Tensor):
		frame = frame.detach().cpu().numpy()
	frame = np.asarray(frame)
	if frame.dtype != np.uint8:
		frame = np.clip(frame, 0, 255).astype(np.uint8)
	return frame


# The two videos, in the order they are reported. Both take (env, obs) so the rollout can
# capture them side by side without caring which is which.
FRAME_SOURCES = {"observation": observation_frame, "render": render_frame}


def available_frame_sources(env) -> dict:
	"""Drop the frame sources this task cannot provide, so the other one is still recorded.

	Not every task has both: a state-only dm_control task has no observation camera, and a
	ManiSkill task built without an image observation mode gets `render_mode=None` and cannot
	render either. Rather than predict that from the config, each source is simply tried once.
	"""
	obs = env.reset()
	sources = {}
	for name, frame_source in FRAME_SOURCES.items():
		try:
			frame_source(env, obs)
		except Exception as error:
			tqdm.write(f"No {name} video: {error}")
			continue
		sources[name] = frame_source
	return sources


def rollout_random_episode(env, max_steps: int, sources: dict, desc: str = "Stepping"):
	"""Step `env` with uniformly random actions until it ends or `max_steps` is reached.

	Returns one list of frames per entry in `sources` (each including the reset state), the
	episode return, and whether the task reported success on the final step -- `None` for tasks
	that have no success criterion, which is not the same as one that has it and did not meet it.
	"""
	obs = env.reset()
	frames = {name: [frame_source(env, obs)] for name, frame_source in sources.items()}
	return_ = 0.0
	success = None

	for _ in tqdm(range(max_steps), desc=desc, leave=False):
		# `rand_act` samples the env's own action space, so it is already the shape and dtype
		# that `step` expects for this task
		obs, reward, terminated, truncated, info = env.step(env.rand_act())
		return_ += float(reward)
		# `in` rather than `.get`, since dm_control tasks return a defaultdict whose missing
		# keys would otherwise materialise as zeros
		if "success" in info:
			success = bool(info["success"])
		for name, frame_source in sources.items():
			frames[name].append(frame_source(env, obs))
		if terminated or truncated:
			break

	return frames, return_, success


@hydra.main(version_base=None, config_path=os.path.join(PROJECT_ROOT, "configs"), config_name="config")
def main(cfg: DictConfig):

	# The task is the only thing instantiated -- nothing here needs the model or the runner
	task: OnlineTaskBase = instantiate(cfg.task)
	env = task.make_env()

	num_episodes = OmegaConf.select(cfg, "num_episodes", default=1)
	max_steps = OmegaConf.select(cfg, "max_steps", default=None) or task.episode_length
	fps = OmegaConf.select(cfg, "video_fps", default=15)
	video_path = OmegaConf.select(cfg, "video_path", default=None)
	stem = os.path.abspath(video_path or f"random_rollout_{task.task_name}")
	stem = os.path.splitext(stem)[0]

	# Make the random actions reproducible for a given seed
	seed = OmegaConf.select(cfg, "seed", default=0)
	torch.manual_seed(seed)
	np.random.seed(seed)
	env.action_space.seed(seed)

	sources = available_frame_sources(env)
	if len(sources) == 0:
		raise RuntimeError("This task can produce neither an observation nor a rendered frame")

	os.makedirs(os.path.dirname(stem), exist_ok=True)
	paths = {name: f"{stem}.{name}.mp4" for name in sources}
	# `macro_block_size=1` keeps each camera's resolution exactly as it was produced; the
	# default (16) silently upscales sizes that are not a multiple of it, e.g. 84 -> 96
	writers = {
		name: imageio.get_writer(path, fps=fps, macro_block_size=1)
		for name, path in paths.items()
	}
	try:
		total_frames = 0
		for episode in range(num_episodes):
			frames, return_, success = rollout_random_episode(
				env, max_steps, sources, desc=f"Episode {episode}"
			)
			for name, writer in writers.items():
				for frame in frames[name]:
					writer.append_data(frame)
			# every source captures on the same steps, so their frame counts agree
			episode_frames = len(frames[next(iter(sources))])
			total_frames += episode_frames
			# `tqdm.write` rather than `print`, so the summary is not overwritten by the step bar
			tqdm.write(
				f"Episode {episode}: {episode_frames} frames, return={return_:.2f}"
				+ (f", success={success}" if success is not None else "")
			)
	finally:
		for writer in writers.values():
			writer.close()

	tqdm.write(f"Wrote {total_frames} frames ({total_frames / fps:.1f}s at {fps} fps) to:")
	for name, path in paths.items():
		tqdm.write(f"  {name}: {path}")


if __name__ == "__main__":
	main()
