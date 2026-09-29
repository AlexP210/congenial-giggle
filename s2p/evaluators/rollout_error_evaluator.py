import typing

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
import tqdm
import os
import matplotlib
matplotlib.use("Agg")
from scipy import stats

import matplotlib.pyplot as plt
from s2p.losses.loss_function_base import LossFunctionBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.models.agent_model import AgentModel

class RolloutErrorEvaluator(OfflineEvaluatorBase):

	def __init__(self, cfg, 
			  task:OfflineTaskBase, 
		):

		super().__init__(cfg, task)
		self.cfg = cfg
		self.best_reward_error = float("inf")
		self.best_dynamics_error = float("inf")
		self.best_value_error = float("inf")

	def __call__(self, model:AgentModel, dataset:Dataset) -> typing.Dict[str, typing.Any]:
				
		n_step_s_predictions = {
			i: []
			for i in range(1,self.cfg.horizon+1)
		}	
		n_step_r_predictions = {
			i: []
			for i in range(1,self.cfg.horizon+1)
		}	
		n_step_r_gt = {
			i: []
			for i in range(1,self.cfg.horizon+1)
		}	
		indices = torch.randint(0, len(dataset), ( min(self.cfg.num_trajectories, len(dataset)), ))
		with torch.no_grad():
			for i in tqdm.tqdm(indices, desc="Rollout Error Evaluation"):
				batch = dataset[i].unsqueeze(0).to(self.cfg.device)

				obs = batch["obs"].permute((1, 0, *range(2, batch["obs"].dim())))
				a = batch["action"].permute((1,0,2))[1:]
				r = batch["reward"].permute((1,0,2))[1:]
				all_s = torch.cat([
					model.encoder_model.encode(obs[i:i + self.cfg.batch_size])
					for i in range(0, obs.shape[0], self.cfg.batch_size)
				])
				s = all_s[:-1]
				sprime = all_s[1:]

				latent_dim = s.shape[-1]

				s_ = s
				sprime_ = sprime
				a_ = a
				r_ = r

				for h in range(1, self.cfg.horizon+1):
					predicted_sprime_ = model.dynamics_model.dynamics(s_, a_)
					predicted_r_ = model.reward_model.reward(s_, a_)

					sprime_error_ = np.clip(
						predicted_sprime_.cpu() - sprime_.cpu(), 
						-self.cfg.max_dynamics_error, self.cfg.max_dynamics_error
					)
					n_step_s_predictions[h].extend(sprime_error_.view(-1,latent_dim).cpu().unbind())

					n_step_r_predictions[h].extend(
						predicted_r_.flatten().cpu().unbind()
					)
					n_step_r_gt[h].extend(
						r_.flatten().cpu().unbind()
					)

					s_ = predicted_sprime_[:-1]
					sprime_ = sprime_[1:]
					a_ = a_[1:]
					r_ = r_[1:]

		info = {}
		# N step latent errors
		# errors: list of (512,) arrays or tensors
		n_plots = self.cfg.horizon
		ncols = max(1, int(np.ceil(np.sqrt(n_plots))))
		nrows = max(1, int(np.ceil(n_plots / ncols)))
		fig, ax = plt.subplots(nrows, ncols, squeeze=False)
		ax = ax.ravel()
		for h in range(1, self.cfg.horizon+1):

			errors = np.stack(n_step_s_predictions[h])  # shape (N, 512)

			# parameters
			num_bins = 50

			# compute global bin edges (consistent across dimensions)
			min_val = errors.min()
			max_val = errors.max()
			bins = np.linspace(min_val, max_val, num_bins + 1)

			# build histogram image
			hist_image = np.zeros((num_bins, errors.shape[1]))

			for d in range(errors.shape[1]):
				hist, _ = np.histogram(errors[:, d], bins=bins)
				hist_image[:, d] = hist

			# optional: normalize per column (so columns are comparable)
			hist_image = hist_image / (hist_image.sum(axis=0, keepdims=True) + 1e-8)
			# plot
			ax[h-1].imshow(
				hist_image,
				aspect='auto',
				origin='lower',
				extent=[0, errors.shape[1], min_val, max_val]
			)
		
		for h in range(n_plots, nrows * ncols):
			ax[h].set_visible(False)
		fig.supxlabel("Dimension of Latent")
		fig.supylabel("Prediction Error")
		plt.tight_layout()
		# Render the figure
		fig.canvas.draw()
		# Convert to numpy array
		img = np.asarray(fig.canvas.buffer_rgba())
		info["dynamics/error_plot"] = img
		plt.close(fig)

		# N step reward errors
		fig, ax = plt.subplots(nrows, ncols, squeeze=False)
		ax = ax.ravel()
		for h in range(1, self.cfg.horizon+1):

			true_r = n_step_r_gt[h]
			predicted_r = n_step_r_predictions[h]
			res = stats.pearsonr(predicted_r, true_r)
			info[f"reward/{h}_step_correlation"] = res.statistic
			ax[h-1].scatter(true_r, predicted_r, s=1, c="k")

		for h in range(n_plots, nrows * ncols):
			ax[h].set_visible(False)
		fig.supxlabel("True Rewards")
		fig.supylabel("Predicted Rewards")
		plt.tight_layout()
		# Render the figure
		fig.canvas.draw()
		# Convert to numpy array
		img = np.asarray(fig.canvas.buffer_rgba())
		info["reward/error_plot"] = img
		plt.close(fig)
		return info
