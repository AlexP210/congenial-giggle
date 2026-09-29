import contextlib
import typing

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
import tqdm
from tensordict import TensorDict

from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.models.agent_model import AgentModel


class SaliencyEvaluator(OfflineEvaluatorBase):
	"""
	Input-gradient saliency along a single trajectory of the dataset, rendered as one video
	per image observation key: the observation with its saliency map overlaid, a frame per
	timestep. For eyeballing what the encoder looks at — not a quantitative measure.

	The scalar being differentiated is `(z * v).sum()` for random unit-normal `v`, averaged
	over `cfg.n_projections` draws. Backpropagating `‖z‖` instead — the obvious choice —
	only measures sensitivity along `z`'s own direction, so pixels the encoder uses to tell
	states apart *without* changing the latent's norm leave no trace in the map. Random
	projections reach every latent direction in expectation, for the same cost per pass.
	"""

	# Weight of the saliency map in the blend over the observation
	OVERLAY_ALPHA = 0.5

	# Quantile of a video's saliency values that its colour scale saturates at
	CLIP_QUANTILE = 0.999

	def __init__(self, cfg,
			  task:OfflineTaskBase,
		):

		super().__init__(cfg, task)
		self.cfg = cfg

	def _sample_trajectory(self, dataset:Dataset) -> typing.List[typing.Union[TensorDict, torch.Tensor]]:
		"""
		Every observation of one trajectory, in order — a frame of the video each.

		The trajectory is drawn off `cfg.seed`, so that every call picks the same one: the
		point is to watch the maps develop over training, not to compare maps of different
		trajectories.
		"""
		if len(dataset) == 0:
			return []

		rng = np.random.default_rng(self.cfg.seed)
		observations = dataset[int(rng.integers(len(dataset)))]["obs"]
		# Skip the first timestep, matching the other offline evaluators
		return [observations[timestep] for timestep in range(1, dataset.horizon)]

	def _prepare_observation(self, observation):
		"""
		Rebuild the observation as gradient-carrying leaves with the (B, T) dims the encoders
		index their shapes off, and report which of them are images.

		A uint8 leaf is an image — the same signal ManiSkillTask uses to tell a camera sensor
		from a float feature map — and is scaled to [0, 1] here. That scaling is not cosmetic:
		the DINOv3 transform's `ToDtype(scale=True)` only rescales when converting *from* an
		integer dtype, so a float image has to arrive already scaled or `Normalize` silently
		sees [0, 255] and the backbone gets garbage. Integer tensors also cannot require grad,
		which is why the conversion has to happen before `requires_grad_` — otherwise the
		result is a non-leaf and no gradient is available for it.

		Returns the model input, the differentiable leaves keyed the same way, and the subset
		of those keys that are images.
		"""
		is_tensordict = isinstance(observation, TensorDict)
		keys = (
			list(observation.keys(include_nested=True, leaves_only=True))
			if is_tensordict else [None]
		)

		prepared, leaves, image_keys = {}, {}, set()
		for key in keys:
			value = (observation[key] if is_tensordict else observation).to(self.cfg.device)
			if value.dtype == torch.uint8:
				value = value.float().div(255.0)
				image_keys.add(key)
			value = value[None, None]  # (B=1, T=1, ...)
			if torch.is_floating_point(value):
				value = value.requires_grad_(True)
				leaves[key] = value
			# Non-float leaves (flags, indices) cannot carry a gradient, but still have to
			# reach the encoder, so they go into the input without becoming a leaf.
			prepared[key] = value

		if not is_tensordict:
			return prepared[None], leaves, image_keys

		model_input = TensorDict({}, batch_size=[1, 1], device=self.cfg.device)
		for key, value in prepared.items():
			model_input[key] = value
		return model_input, leaves, image_keys

	@contextlib.contextmanager
	def _pixel_gradients_enabled(self, model:AgentModel):
		"""
		Open a gradient path from the latent back to the pixels for the duration of the pass.

		The DINOv3 encoders wrap their backbone in `torch.no_grad()` when frozen, and the
		passthrough variant short-circuits to cached `dino_patch_features` when the dataset
		carries them (as the preprocessed ManiSkill datasets do). Either one leaves the input
		disconnected from the latent, so every gradient would come back `None`.
		"""
		overridden = [m for m in model.modules() if hasattr(m, "force_pixel_gradients")]
		for module in overridden:
			module.force_pixel_gradients = True
		try:
			yield
		finally:
			for module in overridden:
				module.force_pixel_gradients = False

	def _encode(self, encoder, observation) -> torch.Tensor:
		"""
		The stochastic encoders' `encode` returns an `rsample`, which would make the map
		depend on the reparameterisation noise as much as on the pixels — the same frame
		gives a different map on every call. Use the distribution's mean where there is one.
		"""
		encode_distribution = getattr(encoder, "encode_distribution", None)
		if encode_distribution is None:
			return encoder.encode(observation)
		return encode_distribution(observation).mean

	def _latest_frame(self, image:torch.Tensor) -> torch.Tensor:
		"""
		(..., C, H, W) -> (C, H, W) for the most recent frame of the observation's stack.

		The `num_frames` entries of one stack are consecutive frames of the same trajectory
		and look nearly identical, so the video gives a frame to each *observation* along the
		trajectory and shows only its newest frame.
		"""
		return image.detach().reshape(-1, *image.shape[-3:])[-1]

	def _saliency_map(self, gradient:torch.Tensor) -> torch.Tensor:
		"""
		(..., C, H, W) gradient -> (H, W) map for the frame `_latest_frame` displays.
		Magnitude, maxed over colour channels because a pixel matters if *any* of its
		channels does. Left unnormalised: the video scales all of its frames together.
		"""
		magnitude = gradient.abs().amax(dim=-3)
		return magnitude.reshape(-1, *magnitude.shape[-2:])[-1]

	def _colour_scale(self, saliencies:typing.List[torch.Tensor]) -> float:
		"""
		The saliency value at which the colormap saturates, shared by every frame of one video.

		`CLIP_QUANTILE` of the pooled values rather than the largest one: input gradients are
		heavy-tailed, so a single hot pixel at a single timestep would otherwise set the scale
		for the whole trajectory and crush every other frame into the bottom of the colormap.
		Pooling the frames — rather than taking a quantile per frame — keeps the single scale
		that makes brightness comparable between timesteps; anything above it clips.
		"""
		pooled = torch.cat([saliency.flatten() for saliency in saliencies]).float().cpu().numpy()
		scale = float(np.quantile(pooled, self.CLIP_QUANTILE))
		# A map that is zero across more than `CLIP_QUANTILE` of its pixels — a very sparse
		# saliency, or an encoder that barely looks at the image — has a zero quantile, which
		# would saturate every pixel that *is* nonzero. The peak is the honest scale there.
		if scale <= 0.0:
			scale = float(pooled.max())
		return max(scale, torch.finfo(saliencies[0].dtype).eps)

	def _render(self, images:typing.List[torch.Tensor], saliencies:typing.List[torch.Tensor]) -> np.ndarray:
		"""
		The trajectory as a (T, C, H, W) uint8 video, each observation blended with its
		saliency map — the layout `_log_wandb` turns into a `wandb.Video`.

		The frames share the colour scale `_colour_scale` picks, so brightness can be compared
		between timesteps. That scale is per-video and therefore arbitrary across evaluations
		— `{key}_gradient_magnitude` is the comparable number.
		"""
		scale = self._colour_scale(saliencies)
		colormap = matplotlib.colormaps["inferno"]

		frames = []
		for image, saliency in zip(images, saliencies):
			# (H, W, C), so a single-channel frame broadcasts against the RGB heatmap
			frame = image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
			heatmap = colormap((saliency.cpu().numpy() / scale).clip(0.0, 1.0))[..., :3]
			blended = (1.0 - self.OVERLAY_ALPHA) * frame + self.OVERLAY_ALPHA * heatmap
			frames.append((blended * 255).astype(np.uint8).transpose(2, 0, 1))
		return np.stack(frames)

	@staticmethod
	def _key_name(key) -> str:
		if key is None:
			return "obs"
		return "/".join(key) if isinstance(key, tuple) else key

	def _observation_gradients(self, model:AgentModel, observation):
		"""
		Mean |d latent / d observation| over `cfg.n_projections` random latent directions, for
		one observation. Returns the prepared input, that mean per differentiable key, and
		which of those keys are images.
		"""
		model_input, leaves, image_keys = self._prepare_observation(observation)
		keys = list(leaves)

		# Accumulate |gradient| per projection rather than summing the gradients themselves,
		# whose signs would cancel across draws. `torch.autograd.grad` keeps this out of the
		# leaves' `.grad`, so there is nothing to zero between passes.
		totals = {key: torch.zeros_like(leaves[key]) for key in keys}
		used = set()

		# Seeded per observation, so every timestep is projected onto the same latent
		# directions and differences between the video's frames come from the observations
		# rather than from the draw.
		generator = torch.Generator(device=self.cfg.device).manual_seed(self.cfg.seed)
		with self._pixel_gradients_enabled(model), torch.enable_grad():
			latent = self._encode(model.encoder_model, model_input)
			for _ in range(self.cfg.n_projections):
				projection = torch.randn(
					latent.shape, generator=generator, device=latent.device, dtype=latent.dtype
				)
				gradients = torch.autograd.grad(
					(latent * projection).sum(),
					[leaves[key] for key in keys],
					retain_graph=True,
					# Observation keys the encoder does not consume (poses the task exposes for
					# the state/probe losses, cached features it now recomputes) have no path to
					# the latent at all, and would otherwise raise here.
					allow_unused=True,
				)
				for key, gradient in zip(keys, gradients):
					if gradient is not None:
						totals[key] += gradient.abs()
						used.add(key)

		magnitudes = {key: totals[key] / self.cfg.n_projections for key in used}
		return model_input, magnitudes, image_keys

	def __call__(self, model:AgentModel, dataset:Dataset) -> typing.Dict[str, typing.Any]:

		# One observation per pass rather than one batched pass: this is a visual, and keeping
		# the timesteps separate means each map is independent of the others by construction.
		# `cfg.n_projections` backward passes per observation is the whole cost of the
		# evaluator — the blending in `_render` is negligible beside it — so the bar tracks
		# observations and reaching the end of it means the videos are done.
		passes = [
			self._observation_gradients(model, observation)
			for observation in tqdm.tqdm(
				self._sample_trajectory(dataset), desc="Saliency Evaluation"
			)
		]

		# Nothing to look at yet (e.g. the first evaluation of an online run, before the
		# buffer has anything in it)
		if not passes:
			return {}

		image_keys = set.union(*(image_keys for _, _, image_keys in passes))
		used = set.union(*(set(magnitudes) for _, magnitudes, _ in passes))

		unused_images = image_keys - used
		if unused_images:
			raise RuntimeError(
				f"No gradient reached the image observation(s) "
				f"{sorted(self._key_name(key) for key in unused_images)}, so there is no "
				f"saliency to show. The encoder has no differentiable path back to the pixels "
				f"— check that the image key is one the encoder under evaluation consumes."
			)

		info = {}
		for key in used:
			name = self._key_name(key)
			key_passes = [
				(model_input, magnitudes[key])
				for model_input, magnitudes, _ in passes if key in magnitudes
			]
			info[f"{name}_gradient_magnitude"] = float(
				np.mean([float(magnitude.sum()) for _, magnitude in key_passes])
			)
			if key in image_keys:
				images = [
					self._latest_frame(model_input[key] if isinstance(model_input, TensorDict) else model_input)
					for model_input, _ in key_passes
				]
				video = self._render(
					images, [self._saliency_map(magnitude) for _, magnitude in key_passes]
				)
				info[f"{name}_saliency"] = video
		return info
