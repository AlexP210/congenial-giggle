import contextlib
import math
import typing

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
import tqdm
from tensordict import TensorDict

from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.models.agent_model import AgentModel
from s2p.models.teacher_student_encoder_model import TeacherStudentEncoderModel
from s2p.lib.stochastic_models import StochasticCAP
from s2p.lib.deterministic_models import DeterministicCAP


class AttentionEvaluator(OfflineEvaluatorBase):
	"""
	The cross-attention a StochasticCAP student pays to its teacher's patch tokens along a
	single trajectory of the dataset, rendered as one video per query: the observation with
	that query's attention over the patch grid overlaid, a frame per timestep.

	Answers the same question as SaliencyEvaluator — what does the encoder look at — from the
	opposite direction. Saliency perturbs the input and watches the latent, so it measures the
	whole encoder including the teacher's own receptive fields, and it costs a backward pass
	per projection. This reads the attention the student's pooling *already computes* on the
	forward pass, so it is nearly free, but it only sees the last step of the pipeline: it
	shows which teacher tokens the latent is read from, not what inside a token the teacher
	looked at. The two disagreeing is informative rather than a bug — a query attending to a
	patch whose pixels carry no gradient means the token, not the patch, is what matters.

	One video per query, because that is the question attention answers and saliency cannot:
	whether the `n_queries` learnable queries have specialised on different parts of the scene
	or collapsed onto the same patches.
	"""

	# Weight of the attention map in the blend over the observation
	OVERLAY_ALPHA = 0.5

	# Quantile of a video's attention values that its colour scale saturates at
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
		trajectories. Sharing the seed with SaliencyEvaluator also means both evaluators walk
		the *same* trajectory, so their videos can be put side by side.
		"""
		if len(dataset) == 0:
			return []

		rng = np.random.default_rng(self.cfg.seed)
		observations = dataset[int(rng.integers(len(dataset)))]["obs"]
		# Skip the first timestep, matching the other offline evaluators
		return [observations[timestep] for timestep in range(1, dataset.horizon)]

	def _prepare_observation(self, observation):
		"""
		Rebuild the observation with the (B, T) dims the encoders index their shapes off, and
		report which of its leaves are images.

		A uint8 leaf is an image — the same signal ManiSkillTask uses to tell a camera sensor
		from a float feature map — and is scaled to [0, 1] here. That scaling is not cosmetic:
		the DINOv3 transform's `ToDtype(scale=True)` only rescales when converting *from* an
		integer dtype, so a float image has to arrive already scaled or `Normalize` silently
		sees [0, 255] and the backbone gets garbage.
		"""
		is_tensordict = isinstance(observation, TensorDict)
		keys = (
			list(observation.keys(include_nested=True, leaves_only=True))
			if is_tensordict else [None]
		)

		prepared, image_keys = {}, set()
		for key in keys:
			value = (observation[key] if is_tensordict else observation).to(self.cfg.device)
			if value.dtype == torch.uint8:
				value = value.float().div(255.0)
				image_keys.add(key)
			prepared[key] = value[None, None]  # (B=1, T=1, ...)

		if not is_tensordict:
			return prepared[None], image_keys

		model_input = TensorDict({}, batch_size=[1, 1], device=self.cfg.device)
		for key, value in prepared.items():
			model_input[key] = value
		return model_input, image_keys

	@staticmethod
	def _key_name(key) -> str:
		if key is None:
			return "obs"
		return "/".join(key) if isinstance(key, tuple) else key

	def _image_key(self, encoder_model:TeacherStudentEncoderModel, image_keys:set):
		"""
		The observation key whose patches the teacher's tokens came from — the pixels the maps
		are laid over.

		The vision encoders (DINOv3 and its passthrough) each record the key they read, so ask
		them rather than inferring it; a single image observation needs no disambiguation and is
		taken as it comes. Anything else is ambiguous, and guessing would overlay one camera's
		attention on another camera's pixels without ever looking wrong.
		"""
		vision_keys = {
			module.cfg.observation_key
			for module in encoder_model.teacher_encoder.modules()
			if hasattr(module, "num_tokens")
			and getattr(module.cfg, "observation_key", None) is not None
		}
		candidates = (vision_keys & image_keys) or vision_keys or image_keys
		if len(candidates) != 1:
			raise RuntimeError(
				f"Cannot tell which observation the teacher's patch tokens belong to: the "
				f"teacher's vision encoders read {sorted(map(str, vision_keys)) or 'no declared key'} "
				f"and the observation's image keys are {sorted(map(str, image_keys)) or 'none'}. "
				f"AttentionEvaluator needs exactly one image to lay its maps over."
			)
		return next(iter(candidates))

	def _patch_grid(self, num_tokens:int) -> typing.Tuple[int, int]:
		"""
		The (rows, columns) patch grid that `num_tokens` tokens of one frame unflatten to.

		Square, because the DINOv3 transform resizes to `(resize_size, resize_size)` before
		patching, whatever the observation's aspect ratio — so the grid is always
		`resize_size // patch_size` on a side, and stretching it back onto the observation's own
		(H, W) is the exact inverse of that resize.
		"""
		side = math.isqrt(num_tokens)
		if side < 2 or side * side != num_tokens:
			raise RuntimeError(
				f"The teacher returns {num_tokens} token(s) per frame, which is not a square "
				f"number greater than one and so cannot be laid out as a patch grid. "
				f"AttentionEvaluator needs a teacher whose tokens *are* image patches "
				f"(DINOv3 with `token_mode: patch`); `cls` and `pool` collapse the patches into "
				f"a single token, leaving no spatial layout to overlay."
			)
		return side, side

	@contextlib.contextmanager
	def _captured_attention(self, cap:StochasticCAP, captured:typing.List[torch.Tensor]):
		"""
		Append every CAP block's cross-attention weights to `captured` for the duration of the
		pass, as (B, n_queries, n_tokens).

		`_CAPBlock.forward` throws the weights away (`attn_out, _ = self.cross_attn(q, kv, kv)`),
		but `nn.MultiheadAttention` computes and returns them anyway — `need_weights` defaults
		to True — so a forward hook on the attention module picks them up at no extra cost.
		Recomputing them here instead would mean either duplicating the block's projections or
		re-entering the module from inside its own hook.
		"""
		def hook(module, args, output):
			weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
			if weights is None:
				raise RuntimeError(
					"A CAP block's cross-attention returned no attention weights, so there is "
					"nothing to visualise. `_CAPBlock.forward` must call `self.cross_attn` with "
					"`need_weights=True` (its default) for AttentionEvaluator to observe it."
				)
			captured.append(weights)

		handles = [block.cross_attn.register_forward_hook(hook) for block in cap.blocks]
		try:
			yield
		finally:
			for handle in handles:
				handle.remove()

	def _attention(self, encoder_model:TeacherStudentEncoderModel, model_input):
		"""
		Per-query attention over the teacher's tokens for one observation, as
		(n_queries, n_tokens), together with the token count of a single frame.

		The teacher is run separately from the student rather than through the wrapper's
		`encode` — which is exactly `student(teacher(obs))` — because the teacher's output shape
		is what says how the flat token axis the student attends over splits back into
		(frames, patches).
		"""
		student = encoder_model.student_encoder
		captured = []

		# No gradients anywhere: unlike saliency, attention is read off the forward pass, so
		# there is nothing to backpropagate and no reason to build a graph. This also leaves
		# the passthrough teacher free to serve the dataset's cached `dino_patch_features`,
		# which are the very tokens the student was distilled against.
		with torch.no_grad(), self._captured_attention(student.encoder, captured):
			tokens = encoder_model.teacher_encoder.encode(model_input)
			student.encode(tokens)

		if not captured:
			raise RuntimeError(
				"The student's cross-attention never ran, so there is no attention to show. "
				"`student_encoder.encoder.blocks` is empty — check `cfg.n_blocks`."
			)

		# Every block cross-attends over the same teacher tokens (the queries never attend to
		# each other), so a block's map is directly comparable to the next one's and the mean
		# over the stack reads as "where query i draws from". With the usual `n_blocks: 1` it
		# is that one block's map exactly.
		attention = torch.stack([weights[0] for weights in captured]).mean(dim=0)

		# The student flattens every dim between the batch dims and the token length into the
		# token axis, so this is the invariant that lets the axis be split back up.
		num_tokens = tokens.shape[-2]
		sequence = math.prod(tokens.shape[2:-1])
		if attention.shape[-1] != sequence:
			raise RuntimeError(
				f"The student attends over {attention.shape[-1]} tokens but the teacher returned "
				f"{sequence} ({tuple(tokens.shape)}), so the attention axis cannot be laid back "
				f"out over the teacher's tokens."
			)
		return attention, num_tokens

	def _latest_frame(self, image:torch.Tensor) -> torch.Tensor:
		"""
		(..., C, H, W) -> (C, H, W) for the most recent frame of the observation's stack.

		The `num_frames` entries of one stack are consecutive frames of the same trajectory and
		look nearly identical, so the video gives a frame to each *observation* along the
		trajectory and shows only its newest frame.
		"""
		return image.detach().reshape(-1, *image.shape[-3:])[-1]

	def _colour_scale(self, maps:typing.List[torch.Tensor]) -> float:
		"""
		The attention weight at which the colormap saturates, shared by every frame of one
		video.

		`CLIP_QUANTILE` of the pooled values rather than the largest one: attention is often
		near-degenerate early in training, and a single patch spiking at a single timestep would
		otherwise set the scale for the whole trajectory and crush every other frame into the
		bottom of the colormap. Pooling the frames — rather than taking a quantile per frame —
		keeps the single scale that makes brightness comparable between timesteps; anything
		above it clips.
		"""
		pooled = torch.cat([attention.flatten() for attention in maps]).float().cpu().numpy()
		scale = float(np.quantile(pooled, self.CLIP_QUANTILE))
		# A query attending to fewer than `1 - CLIP_QUANTILE` of its tokens has a zero
		# quantile, which would saturate every patch it *does* attend to. The peak is the
		# honest scale there.
		if scale <= 0.0:
			scale = float(pooled.max())
		return max(scale, torch.finfo(maps[0].dtype).eps)

	def _render(self, images:typing.List[torch.Tensor], maps:typing.List[torch.Tensor]) -> np.ndarray:
		"""
		One query's trajectory as a (T, C, H, W) uint8 video, each observation blended with that
		query's attention over the patch grid — the layout `_log_wandb` turns into a
		`wandb.Video`.

		Patches are upsampled with `nearest`, so each one stays a visible block: the attention
		has no structure inside a patch, and smoothing it would imply a resolution it does not
		have. The frames share the colour scale `_colour_scale` picks, so brightness can be
		compared between timesteps — but that scale is per-video, and each query gets its own,
		so brightness cannot be compared between queries. `{name}_attention_entropy_query{i}`
		is the number that can.
		"""
		scale = self._colour_scale(maps)
		colormap = matplotlib.colormaps["inferno"]

		frames = []
		for image, attention in zip(images, maps):
			# (H, W, C), so a single-channel frame broadcasts against the RGB heatmap
			frame = image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
			patches = F.interpolate(
				attention[None, None].float(), size=frame.shape[:2], mode="nearest"
			)[0, 0]
			heatmap = colormap((patches.cpu().numpy() / scale).clip(0.0, 1.0))[..., :3]
			blended = (1.0 - self.OVERLAY_ALPHA) * frame + self.OVERLAY_ALPHA * heatmap
			frames.append((blended * 255).astype(np.uint8).transpose(2, 0, 1))
		return np.stack(frames)

	def _assert_supported(self, model:AgentModel) -> TeacherStudentEncoderModel:
		"""
		The encoder this evaluator can read: a teacher-student pair whose student pools the
		teacher's tokens with cross-attention. Everything else has no per-query attention to
		show — use SaliencyEvaluator, which works on any encoder.
		"""
		encoder_model = model.encoder_model
		assert isinstance(encoder_model, TeacherStudentEncoderModel), (
			f"AttentionEvaluator needs a TeacherStudentEncoderModel to read the student's "
			f"attention over the teacher's tokens, got {type(encoder_model).__name__}."
		)
		student = getattr(encoder_model.student_encoder, "encoder", None)
		assert isinstance(student, StochasticCAP) or isinstance(student, DeterministicCAP), (
			f"AttentionEvaluator needs a student whose `encoder` is a StochasticCAP — the "
			f"cross-attention pooling is what it visualises — got "
			f"{type(student).__name__} on {type(encoder_model.student_encoder).__name__}."
		)
		return encoder_model

	def __call__(self, model:AgentModel, dataset:Dataset) -> typing.Dict[str, typing.Any]:

		encoder_model = self._assert_supported(model)

		observations = self._sample_trajectory(dataset)
		# Nothing to look at yet (e.g. the first evaluation of an online run, before the
		# buffer has anything in it)
		if not observations:
			return {}

		# One observation per pass rather than one batched pass, matching SaliencyEvaluator: the
		# frames of the video stay independent of each other by construction. A pass is a single
		# forward with no backward, so the bar moves fast — it is here because the teacher is
		# the expensive part of it whenever the dataset carries no cached features.
		key, grid, num_tokens = None, None, None
		images, attentions = [], []
		for observation in tqdm.tqdm(observations, desc="Attention Evaluation"):
			model_input, image_keys = self._prepare_observation(observation)
			attention, tokens_per_frame = self._attention(encoder_model, model_input)

			if key is None:
				key = self._image_key(encoder_model, image_keys)
				grid = self._patch_grid(tokens_per_frame)
				num_tokens = tokens_per_frame
				queries = encoder_model.student_encoder.encoder.queries
				assert attention.shape[0] == queries.shape[1], (
					f"The student has {queries.shape[1]} queries but its attention came back "
					f"with {attention.shape[0]} rows; the maps would not line up with "
					f"`student_encoder.encoder.queries`."
				)

			images.append(
				self._latest_frame(model_input[key] if isinstance(model_input, TensorDict) else model_input)
			)
			attentions.append(attention)

		name = self._key_name(key)
		attention = torch.stack(attentions)                            # (T, n_queries, n_tokens)
		# The token axis is (frames, patches) — the frame stack the teacher encoded, flattened.
		# Only the newest frame's patches get a map, because that is the frame `_latest_frame`
		# displays; `newest_frame_fraction` below is what says whether that frame is even where
		# the query is looking.
		per_frame = attention.unflatten(-1, (-1, num_tokens))          # (T, n_queries, S, N)
		newest = per_frame[:, :, -1]                                   # (T, n_queries, N)

		info = {}
		for query in range(attention.shape[1]):
			weights = attention[:, query]                              # (T, n_tokens)

			# Normalised by log(n_tokens), so 1.0 is attention spread evenly over every token
			# and 0.0 is all of it on one. Unlike the videos' colour scale this is comparable
			# across queries, across evaluations and across runs: it is how a query collapsing
			# (or never specialising) shows up as a number.
			entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
			info[f"{name}_attention_entropy_query{query}"] = float(
				(entropy / math.log(weights.shape[-1])).mean()
			)
			if per_frame.shape[2] > 1:
				info[f"{name}_attention_newest_frame_fraction_query{query}"] = float(
					newest[:, query].sum(-1).mean()
				)

			info[f"{name}_attention_query{query}"] = self._render(
				images, list(newest[:, query].reshape(-1, *grid))
			)
		return info
