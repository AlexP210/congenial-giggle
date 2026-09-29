import random
import typing

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from s2p.evaluators.evaluator_group import EvaluatorGroup
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
from s2p.models.agent_model import AgentModel
from s2p.tasks.base.offline_task_base import OfflineTaskBase


class SnapshotDataset(Dataset):
	"""A fixed list of windows, in the shape an offline evaluator expects to index.

	Deliberately dumb, and a `Dataset` rather than a plain list for one reason:
	`RolloutErrorEvaluator` draws its indices with `torch.randint` and indexes with the
	resulting 0-dim tensors, which a list rejects.
	"""

	def __init__(self, windows:typing.List):
		self.windows = windows

	def __len__(self):
		return len(self.windows)

	def __getitem__(self, index):
		return self.windows[int(index)]


class InitialDataWrapper(OfflineEvaluatorBase):
	"""
	Runs offline evaluators against the data a run *started* with, forever.

	The mirror of `OnlineWrapper`, which swaps the dataset for freshly collected on-policy data
	at every call: this one swaps it for a snapshot taken at the first call and never updated.
	Same purpose in both cases -- fix what the wrapped evaluators see, so that a change in their
	numbers is attributable to the model rather than to the data.

	`evaluators` is a list, a name -> evaluator mapping, or a single evaluator; see
	`EvaluatorGroup`, which also settles how their outputs are named. They share the one
	snapshot, which is the point of taking more than one: the snapshot is resident for the whole
	run, so scoring the same early windows two ways should cost one copy of them, not two -- and
	two snapshots would not even be the same windows, `capture` sampling its indices at random.

	It exists because online training reports everything on a moving target. The buffer holds
	whatever the current policy visits, so a loss that rises can mean the model got worse or that
	the data got harder, and the training curve cannot separate those. Holding the distribution
	fixed does separate them. Read alongside the live metrics:

	  * both flat, or both improving: nothing is being lost; the model is fitting a distribution
	    that keeps moving.
	  * this one degrading while the live metric holds: the model is forgetting the early
	    distribution to fit the current one. With a circular buffer that is the ingredient for a
	    visitation cycle -- planner improves, buffer evicts the states that taught it, planner
	    regresses onto them -- which shows up as an unstable success rate over a settled loss.
	  * this one flat while the live metric degrades: the early data was the *easy* data. Check
	    `reward_true_std` in the training log; a seed of random-action rollouts has roughly a
	    tenth of expert data's reward spread on this project's PushCube recording, so the same
	    model scores far better on it for reasons unrelated to what it knows.

	The snapshot is materialized and cloned, not referenced. During online training the dataset
	handed in is `OnlineTrainer.validation_dataset`, which *is* the replay buffer, and
	`OnlineTransitionDataset.__getitem__` indexes into its backing storage -- so a reference, or
	an uncloned window, would be overwritten as the buffer wrapped around. The metric would
	quietly become "loss on current data" and would look perfectly healthy while measuring
	nothing.

	The first call has nothing to report; it is the one taking the snapshot.
	"""

	def __init__(self, cfg:OmegaConf, task:OfflineTaskBase, evaluators):

		super().__init__(cfg, task)
		self.cfg = cfg
		self.evaluators = EvaluatorGroup(evaluators)
		# None until the first call; a `SnapshotDataset` afterwards.
		self.snapshot = None

	def set_save_path(self, filepath):
		# Deliberately *not* forwarded to the wrapped evaluators, which is where this differs from
		# `OnlineWrapper`. An evaluator that checkpoints on its own best score (`LossEvaluator`)
		# would here be selecting the model that forgot the least, which is not a checkpoint
		# anyone wants restored -- and it writes a full agent each time it improves. Leaving the
		# wrapped evaluators' save paths unset makes `_save_best_checkpoint` a no-op.
		super().set_save_path(filepath)

	def capture(self, dataset) -> SnapshotDataset:
		"""Copy `cfg.fraction` of `dataset`'s windows out of it, at most `cfg.max_windows`.

		`max_windows` is the memory bound, and it is a real one: the snapshot is resident for the
		whole run, and one window is `horizon` steps of a frame stack -- ~1.8 MB at this
		project's PushCube settings (horizon 4, 3 frames, 224x224 uint8).
		"""
		n = len(dataset)
		k = min(max(1, int(n * self.cfg.fraction)), int(self.cfg.max_windows), n)
		indices = random.sample(range(n), k)
		return SnapshotDataset([
			dataset[index].clone().to(self.cfg.storage_device) for index in indices
		])

	def __call__(self, model:AgentModel, dataset) -> typing.Dict[str, typing.Any]:

		if self.snapshot is None:
			self.snapshot = self.capture(dataset)
			print(
				f"[InitialDataWrapper] captured {len(self.snapshot)} windows of the initial data "
				f"distribution for {', '.join(self.evaluators.names)}; reporting from the next call on."
			)
			return {}

		return self.evaluators(model, self.snapshot)
