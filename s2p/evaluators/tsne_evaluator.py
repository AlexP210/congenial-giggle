import typing

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tensordict import TensorDict
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
import tqdm

from s2p.losses.loss_function_base import LossFunctionBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.models.agent_model import AgentModel

class TSNEEvaluator(OfflineEvaluatorBase):

	def __init__(self, cfg, task:OfflineTaskBase):

		super().__init__(cfg, task)
		self.cfg = cfg

	def _gather(self, dataset:Dataset, key:str, n_indices, t_indices) -> torch.Tensor:
		"""
		Pick out one timestep per sampled trajectory as a single gather on the
		dataset's flat storage.

		Going through `dataset[n]` instead would build a TensorDict per sample on
		`batch_device`, copying every key for all `horizon` timesteps to that device;
		`[key][t]` is then a view that keeps the whole horizon-sized allocation alive
		until the gather completes, so peak memory is `horizon` times what we need.
		"""
		n_indices = torch.as_tensor(n_indices, dtype=torch.long)
		t_indices = torch.as_tensor(t_indices, dtype=torch.long)

		# Only the TensorDict-backed datasets keep a flat [length, horizon, ...] buffer
		# we can gather from. Note that `.data` is not always that: on
		# ManiSkillTrajectoryDataset it is an open h5py.File keyed by trajectory group,
		# so check the type and batch layout rather than just the attribute's presence.
		storage = getattr(dataset, "data", None)
		if (
			isinstance(storage, TensorDict)
			and len(storage.batch_size) >= 2
			and key in storage.keys()
		):
			return storage[key][n_indices, t_indices]

		# Everything else: per-item indexing, as before. `clone()` (rather than a plain
		# `.cpu()`, which is a no-op for data already on the CPU) is what forces a real
		# copy, so each item's remaining horizon can be freed instead of being kept
		# alive by the view; `.cpu()` then keeps the stack off the device. Both work for
		# a plain tensor and for the nested TensorDict that composite observations
		# (e.g. one key per camera) index down to.
		return torch.stack([
			dataset[int(n)][key][int(t)].clone().cpu()
			for n, t in zip(n_indices, t_indices)
		])

	def __call__(self, model:AgentModel, dataset:Dataset) -> typing.Dict[str, typing.Any]:

		N = len(dataset)
		T = dataset.horizon - 1  # skip first timestep, matching original [:,1:]

		# If called with empty dataset (e.g. first iteration on online training)
		# do nothin

		indices = np.random.choice(N * T, size=min(self.cfg.n_samples, N*T), replace=False)
		n_indices = indices // T
		t_indices = (indices % T) + 1  # +1 to skip first timestep

		observations = self._gather(dataset, "obs", n_indices, t_indices).unsqueeze(1)
		rewards = self._gather(dataset, "reward", n_indices, t_indices).unsqueeze(1)

		# Move one batch at a time so only `batch_size` observations are ever resident
		# on the device, rather than all `n_samples` of them.
		with torch.no_grad():
			latents_np = np.concatenate([
				model.encoder_model.encode(
					observations[i:i + self.cfg.batch_size].to(self.cfg.device)
				).squeeze(1).cpu().numpy()
				for i in range(0, len(observations), self.cfg.batch_size)
			])
		rewards_np = rewards.squeeze(1).cpu().numpy()

		tsne = TSNE(n_components=2, perplexity=self.cfg.perplexity, random_state=self.cfg.seed)
		
		latents_2d = tsne.fit_transform(latents_np)
		# Plot with color by reward
		fig, ax = plt.subplots(1, 1, figsize=(10, 8))
		scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=rewards_np, cmap='viridis', alpha=0.5)
		plt.colorbar(scatter, label='Reward')
		plt.title('t-SNE of Encoded Latents (Colored by Reward)')
		plt.xlabel('t-SNE 1')
		plt.ylabel('t-SNE 2')
		plt.tight_layout()
		# Render the figure
		fig.canvas.draw()
		# Convert to numpy array. Copy, because the array is a view onto the canvas
		# buffer, which is no longer valid once the figure is closed.
		img = np.asarray(fig.canvas.buffer_rgba()).copy()
		# pyplot keeps a global reference to every figure it creates, so without this
		# each evaluation interval leaks a figure plus its canvas buffer.
		plt.close(fig)
		# plt.show()
		return {
			"plot": img
		}
