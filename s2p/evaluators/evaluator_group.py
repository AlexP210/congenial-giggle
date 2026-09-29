import os
import re
import typing

from omegaconf import DictConfig, ListConfig

from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase


class EvaluatorGroup:
	"""
	The offline evaluators a dataset-swapping wrapper runs, and the names their outputs go under.

	`OnlineWrapper` and `InitialDataWrapper` both exist to hand a *different* dataset to an
	offline evaluator, and in both the dataset is the expensive half -- a rollout of
	`num_transitions` on-policy steps for the one, a set of windows resident for the whole run
	for the other. Measuring two things about the same data should not mean collecting or
	holding it twice, so both wrappers take a collection of evaluators rather than a single one
	and run all of them over the one dataset. This is that collection.

	It accepts, from a config:

	  * a list -- `evaluators: [${evaluators.loss}, ${evaluators.rollout_error}]` -- with each
	    entry named after its class (`LossEvaluator` -> `loss`); a repeated class gets a `_2`,
	    `_3`, ... suffix, which is legal but unreadable, so name those explicitly instead;
	  * a mapping -- `evaluators: {loss: ${evaluators.loss}}` -- which names them explicitly and
	    is the only form Hydra's defaults list can compose into, a list element not being
	    addressable as a package (`- /evaluators@evaluators.loss: loss`);
	  * a single evaluator, which is read as a one-element list.

	The outputs are merged into one dict. With more than one evaluator every key is prefixed
	with its evaluator's name (`loss/total_loss`), since two evaluators of the same kind would
	otherwise overwrite each other in the log; with one they are passed through untouched, so a
	wrapper configured the way all of them were before reports exactly the keys it did before.
	The runner prefixes the result again with the wrapper's own name, so a metric arrives in the
	log as `<wrapper>/<evaluator>/<metric>`.
	"""

	def __init__(self, evaluators):
		self.evaluators = self._as_named(evaluators)
		if not self.evaluators:
			raise ValueError(
				"An evaluator wrapper was configured with no evaluators to run; it collects a "
				"dataset for them, so with none there is nothing for it to do."
			)

	def __len__(self) -> int:
		return len(self.evaluators)

	@property
	def names(self) -> typing.List[str]:
		return list(self.evaluators.keys())

	@property
	def qualify_keys(self) -> bool:
		"""Whether output keys and save paths are qualified by the evaluator's name."""
		return len(self.evaluators) > 1

	@staticmethod
	def _default_name(evaluator) -> str:
		"""`RolloutErrorEvaluator` -> `rollout_error`: the class name, minus the suffix every one shares."""
		name = type(evaluator).__name__
		# Split on the case boundaries, the second pattern keeping acronyms whole
		# (`TSNEEvaluator` -> `tsne_evaluator` rather than `t_s_n_e_evaluator`).
		name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
		name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
		return name[: -len("_evaluator")] if name.endswith("_evaluator") else name

	@classmethod
	def _as_named(cls, evaluators) -> typing.Dict[str, OfflineEvaluatorBase]:
		"""Normalize whatever the config produced into an ordered {name: evaluator} mapping."""
		if isinstance(evaluators, OfflineEvaluatorBase):
			entries = [(None, evaluators)]
		elif isinstance(evaluators, (DictConfig, typing.Mapping)):
			entries = [(str(name), evaluator) for name, evaluator in evaluators.items()]
		elif isinstance(evaluators, (ListConfig, list, tuple)):
			entries = [(None, evaluator) for evaluator in evaluators]
		else:
			raise TypeError(
				f"Expected a list, a mapping, or a single evaluator, got {type(evaluators).__name__}."
			)

		named = {}
		for position, (name, evaluator) in enumerate(entries):
			if not isinstance(evaluator, OfflineEvaluatorBase):
				label = f"'{name}'" if name is not None else f"at position {position}"
				raise TypeError(
					f"Evaluator {label} is a "
					f"{type(evaluator).__name__}, which is not an OfflineEvaluatorBase. These "
					"wrappers swap the dataset an offline evaluator reads, so an online "
					"evaluator -- which collects its own -- cannot be wrapped."
				)
			if name is None:
				# Named after its class, with an index appended from the second occurrence on so
				# that two evaluators of the same kind do not collapse onto one set of log keys.
				name = base = cls._default_name(evaluator)
				occurrence = 2
				while name in named:
					name = f"{base}_{occurrence}"
					occurrence += 1
			elif name in named:
				raise ValueError(f"Two evaluators were configured under the same name '{name}'.")
			named[name] = evaluator
		return named

	def set_save_path(self, filepath):
		"""
		Point each evaluator at where it may write its own artifacts.

		A subdirectory per evaluator once there is more than one: `_save_best_checkpoint` names
		its folder after the metric (`best_total_loss`), so two evaluators of the same kind
		sharing a path would write into each other's checkpoint.
		"""
		for name, evaluator in self.evaluators.items():
			if filepath is None or not self.qualify_keys:
				evaluator.set_save_path(filepath)
			else:
				evaluator.set_save_path(os.path.join(filepath, name))

	def __call__(self, model, dataset) -> typing.Dict[str, typing.Any]:
		"""Run every evaluator over `dataset`, merging their outputs into one dict."""
		info = {}
		for name, evaluator in self.evaluators.items():
			output = evaluator(model, dataset)
			if not self.qualify_keys:
				info.update(output)
				continue
			for key, value in output.items():
				info[f"{name}/{key}"] = value
		return info
