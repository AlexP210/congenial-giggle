import contextlib
import typing

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from s2p.evaluators.online_evaluator_base import OnlineEvaluatorBase
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.models.agent_model import AgentModel, ActionMode


class PlannerConvergenceEvaluator(OnlineEvaluatorBase):
	"""
	How the planner converges, measured from the task's initial state.

	The environment is reset once and the planner is run from that state `cfg.num_plans`
	times, keeping the per-iteration trace the planner reports (estimated value of the plan,
	width of the action distribution, movement of the action mean, ...). Those traces are
	plotted against planner iteration and reduced to a handful of scalars.

	It answers "does the planner still improve on its last iteration, or is the iteration
	budget already spent?", which is a property of the model rather than of the state, so one
	state is enough - and the initial state is the one state every task is guaranteed to
	have, with no rollout needed to reach it. All `cfg.num_plans` plans share that single
	reset, so the band around each curve is planner sampling noise alone; set `cfg.reset_seed`
	to pin the state itself and make the curves comparable from one call to the next.

	It is deliberately *not* a measure of plan quality: the value on the curve is the model's
	own estimate, so it rises whether or not the model is right.
	"""

	# One panel per entry, over the planner's per-iteration scalar traces. A key the planner
	# does not report is skipped, so a planner tracing a different set of metrics still gets
	# whatever it does report plotted.
	CURVE_PANELS = (
		{
			"title": "Estimated value of the plan",
			"ylabel": "estimated value",
			"keys": ("value_mean", "value_max", "elite_value_mean"),
		},
		{
			"title": "Spread of the estimated values",
			"ylabel": "std over samples",
			"keys": ("value_std", "elite_value_std"),
		},
		{
			# `total_return` is the planner's discounted sum of predicted rewards over the
			# horizon, without the terminal value bootstrap - so the gap between this and
			# the value panel above is exactly what the value function contributes.
			"title": "Predicted reward of the elites",
			"ylabel": "discounted reward, no bootstrap",
			"keys": ("elite_total_return_mean",),
		},
		{
			"title": "Width of the action distribution",
			"ylabel": "action std",
			"keys": ("action_std_mean", "action_std_max"),
		},
		{
			# Log scale: a converging planner's movement decays geometrically, which a
			# linear axis renders as a spike followed by a flat line at zero.
			"title": "Movement of the action mean",
			"ylabel": "RMS shift per iteration",
			"keys": ("action_mean_shift",),
			"yscale": "log",
		},
		{
			"title": "Effective sample size of the elites",
			"ylabel": "fraction of num_elites",
			"keys": ("elite_weight_ess_fraction",),
		},
		{
			"title": "Elites seeded by the policy",
			"ylabel": "fraction of elites",
			"keys": ("policy_elite_fraction",),
		},
	)

	# Traces whose first and last iteration are reported as scalars, so that convergence can
	# be tracked over training without reading the plots
	SUMMARY_KEYS = (
		"elite_value_mean",
		"elite_total_return_mean",
		"action_std_mean",
		"action_mean_shift",
		"elite_weight_ess_fraction",
		"policy_elite_fraction",
	)

	def __init__(self, cfg,
			  task:OnlineTaskBase,
		):

		super().__init__(cfg, task)
		self.cfg = cfg

		self.env = task.make_env(num_envs=self.cfg.num_envs)

	@property
	def num_plans(self) -> int:
		"""Number of plans to run from the sampled starting state."""
		return int(self.cfg.num_plans)

	def _check_model(self, model:AgentModel) -> None:
		"""
		Refuse an agent which cannot plan.

		`ActionMode.PLANNING` is the agent's own verdict on whether it has the pieces a plan
		needs (dynamics and reward); the planner itself is checked separately, since an agent
		can hold those models without one being attached.
		"""
		if model.planner is None:
			raise ValueError(
				"PlannerConvergenceEvaluator needs an agent with a planner attached, but this "
				"agent has `planner=None`. There is no convergence behaviour to measure."
			)
		if ActionMode.PLANNING not in model.action_modes:
			raise ValueError(
				"PlannerConvergenceEvaluator needs an agent which can plan, but this agent's "
				f"action modes are {[mode.name for mode in model.action_modes]}. A planner "
				"needs both a dynamics model and a reward model."
			)

	@contextlib.contextmanager
	def _convergence_trace_enabled(self, model:AgentModel):
		"""
		Turn the planner's convergence trace on for the duration of the block.

		The trace is off by default because collecting it costs plan latency, which the
		control loop pays on every step and `RealTaskEvaluator` turns into a control rate.
		This evaluator is the thing which reads it, so it switches it on for its own plans
		and leaves the planner the way it found it - including when the planner raises.
		"""
		planner = model.planner
		previous = getattr(planner, "collect_convergence_info", False)
		planner.collect_convergence_info = True
		try:
			yield
		finally:
			planner.collect_convergence_info = previous

	def _initial_observation(self, model:AgentModel):
		"""
		The task's initial state: the observation a reset of the environment hands back.

		With `cfg.reset_seed` set, the reset is seeded, so every call plans from the same
		initial state and the curves can be compared across a training run rather than only
		within one call. It is left unset by default because only some of the tasks' envs
		take a seed - ManiSkill and PushT forward one, the dm_control wrappers take no reset
		arguments at all and would raise.

		Left in whatever dtype the env emits (uint8 for images): the encoders normalize
		internally, so converting here would double-scale them.
		"""
		reset_seed = self.cfg.get("reset_seed")
		reset_kwargs = {} if reset_seed is None else {"seed": int(reset_seed)}
		return self.env.reset(**reset_kwargs)

	@torch.no_grad()
	def _plan_once(self, model:AgentModel, observation) -> typing.Dict[str, np.ndarray]:
		"""Plan once from `observation`, returning the planner's info dict as numpy arrays."""
		# (T=1, B, ...), which is the shape the encoders index their observations off. The
		# observation already carries the env axis (B), so only time is added.
		observation = observation.to(self.cfg.device).unsqueeze(0)
		state = model.encoder_model.encode(observation)

		# A zeroed prior, as at the start of an episode. Note the shape: the planner's
		# `mean` starts as this, so it has to carry the horizon, not just the action dim.
		action_prior = torch.zeros(
			size=(model.planner.cfg.horizon, *self.task.action_dimension),
			device=self.cfg.device,
		)

		_, info = model.plan(state, action_prior)
		return {key: value.detach().float().cpu().numpy() for key, value in info.items()}

	def _collect_traces(self, model:AgentModel, observation) -> typing.Dict[str, np.ndarray]:
		"""
		Run `cfg.num_plans` plans from the one reset state, stacking each metric over plans:
		`[num_plans]` for the per-plan scalars, `[num_plans, iterations]` for the
		per-iteration traces and `[num_plans, iterations, ...]` for the profiles.

		Planning is stochastic - the action samples, and the draw which picks the returned
		plan, differ from run to run - so one trace is a noisy view of the convergence
		behaviour. Repeating from a single state separates that sampling noise (the spread
		across plans, drawn as a band) from the convergence itself (the mean curve).
		"""
		with self._convergence_trace_enabled(model):
			traces = [
				self._plan_once(model, observation)
				for _ in tqdm(range(self.num_plans), "Planner Convergence Evaluation")
			]
		# Intersection, so that a key only some plans report cannot fail the stack
		keys = set.intersection(*[set(trace.keys()) for trace in traces])
		return {key: np.stack([trace[key] for trace in traces]) for key in sorted(keys)}

	@staticmethod
	def _mean_and_sem(traces:typing.Dict[str, np.ndarray], key:str, ndim:int):
		"""
		Mean over plans of one trace, with the standard error of that mean.

		`None` when the planner does not report the key, or reports it with a different
		rank than the caller is about to plot (`ndim` counts the stacked plan dimension).
		"""
		values = traces.get(key)
		if values is None or values.ndim != ndim:
			return None, None
		return values.mean(axis=0), values.std(axis=0) / np.sqrt(values.shape[0])

	def _summary_scalars(self, traces:typing.Dict[str, np.ndarray]) -> typing.Dict[str, typing.Any]:
		"""First and last iteration of each summarised trace, plus the value of the plans."""
		info = {}
		for key in self.SUMMARY_KEYS:
			curve, _ = self._mean_and_sem(traces, key, ndim=2)
			if curve is None:
				continue
			info[f"{key}/initial"] = float(curve[0])
			info[f"{key}/final"] = float(curve[-1])
			info[f"{key}/change"] = float(curve[-1] - curve[0])

		# Value and return of the plans which were actually returned, as opposed to of the
		# elite population they were drawn from
		for key in ("value", "total_return"):
			values = traces.get(key)
			if values is None or values.ndim != 1:
				continue
			info[f"plan_{key}"] = float(values.mean())
			if values.shape[0] > 1:
				info[f"plan_{key}_sem"] = float(values.std() / np.sqrt(values.shape[0]))
				info[f"plan_{key}_distribution"] = values.astype(np.float32)

		info.update(self._convergence_scalars(traces))
		return info

	def _convergence_scalars(self, traces:typing.Dict[str, np.ndarray]) -> typing.Dict[str, typing.Any]:
		"""
		How many iterations the planner needs before its plan stops moving.

		A plan counts as settled on the first iteration whose RMS shift of the action mean
		is below `cfg.convergence_tolerance`. A plan which never settles is counted as
		needing the whole iteration budget, so `converged_fraction` is what says whether
		`iterations_to_convergence` means "converged there" or "ran out of iterations".
		"""
		shifts = traces.get("action_mean_shift")
		if shifts is None or shifts.ndim != 2:
			return {}

		num_iterations = shifts.shape[1]
		settled = shifts < float(self.cfg.convergence_tolerance)
		converged = settled.any(axis=1)
		# `argmax` over a boolean row is the first True - but is also 0 for a row with no
		# True at all, hence the explicit fallback to the full budget
		first_settled = np.where(converged, settled.argmax(axis=1) + 1, num_iterations)

		return {
			"iterations_to_convergence": float(first_settled.mean()),
			"converged_fraction": float(converged.mean()),
			"num_iterations": int(num_iterations),
		}

	@staticmethod
	def _render(fig) -> np.ndarray:
		"""Rasterize a figure into the (H, W, 4) array the runners log as an image."""
		fig.tight_layout()
		fig.canvas.draw()
		image = np.asarray(fig.canvas.buffer_rgba())
		plt.close(fig)
		return image

	def _plot_curves(self, traces:typing.Dict[str, np.ndarray]) -> typing.Optional[np.ndarray]:
		"""One panel per group of per-iteration scalar traces, mean over plans with a sem band."""
		panels = [
			panel for panel in self.CURVE_PANELS
			if any(self._mean_and_sem(traces, key, ndim=2)[0] is not None for key in panel["keys"])
		]
		if not panels:
			return None

		ncols = max(1, int(np.ceil(np.sqrt(len(panels)))))
		nrows = max(1, int(np.ceil(len(panels) / ncols)))
		fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(4.0 * ncols, 3.0 * nrows))
		axes = axes.ravel()

		for ax, panel in zip(axes, panels):
			for key in panel["keys"]:
				curve, sem = self._mean_and_sem(traces, key, ndim=2)
				if curve is None:
					continue
				iterations = np.arange(1, len(curve) + 1)
				line, = ax.plot(iterations, curve, marker="o", markersize=3, label=key)
				ax.fill_between(iterations, curve - sem, curve + sem, alpha=0.25, color=line.get_color())
			ax.set_title(panel["title"], fontsize="small")
			ax.set_xlabel("planner iteration", fontsize="x-small")
			ax.set_ylabel(panel["ylabel"], fontsize="x-small")
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			if "yscale" in panel:
				ax.set_yscale(panel["yscale"])
			ax.legend(fontsize="xx-small")

		for ax in axes[len(panels):]:
			ax.set_visible(False)

		fig.suptitle(f"Planner convergence, mean of {self.num_plans} plans from the initial state", fontsize="medium")
		return self._render(fig)

	def _plot_profiles(self, traces:typing.Dict[str, np.ndarray]) -> typing.Optional[np.ndarray]:
		"""
		The two traces which are a vector per iteration rather than a scalar: how the action
		std is distributed over the planning horizon, and where the executed action lands.
		"""
		std_per_step, _ = self._mean_and_sem(traces, "action_std_per_step", ndim=3)
		first_action, _ = self._mean_and_sem(traces, "first_action_mean", ndim=3)
		if std_per_step is None and first_action is None:
			return None

		panels = [profile for profile in (std_per_step, first_action) if profile is not None]
		fig, axes = plt.subplots(1, len(panels), squeeze=False, figsize=(5.0 * len(panels), 3.5))
		axes = axes.ravel()
		panel = 0

		if std_per_step is not None:
			ax = axes[panel]
			num_iterations, horizon = std_per_step.shape
			image = ax.imshow(
				std_per_step,
				aspect="auto",
				origin="lower",
				extent=[0.5, horizon + 0.5, 0.5, num_iterations + 0.5],
			)
			fig.colorbar(image, ax=ax)
			ax.set_title("Action std over the planning horizon", fontsize="small")
			ax.set_xlabel("planning step", fontsize="x-small")
			ax.set_ylabel("planner iteration", fontsize="x-small")
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			panel += 1

		if first_action is not None:
			ax = axes[panel]
			iterations = np.arange(1, first_action.shape[0] + 1)
			for dimension in range(first_action.shape[1]):
				ax.plot(iterations, first_action[:, dimension], marker="o", markersize=3, label=f"a[{dimension}]")
			ax.set_title("Mean of the executed action", fontsize="small")
			ax.set_xlabel("planner iteration", fontsize="x-small")
			ax.set_ylabel("action", fontsize="x-small")
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			ax.legend(fontsize="xx-small")

		return self._render(fig)

	def __call__(self, model:AgentModel) -> typing.Dict[str, typing.Any]:

		self._check_model(model)

		observation = self._initial_observation(model)

		traces = self._collect_traces(model, observation)

		# A planner which reports only per-plan scalars has nothing to say about its
		# convergence, and would quietly log three numbers and no curves
		if not any(values.ndim >= 2 for values in traces.values()):
			raise ValueError(
				f"{type(model.planner).__name__}.plan reported no per-iteration metrics, so "
				"there is no convergence behaviour to measure. A planner has to fill its info "
				"dict with one entry per planner iteration (see `MPPIPlanner.plan`) and honour "
				"`collect_convergence_info`."
			)

		info = self._summary_scalars(traces)

		curves = self._plot_curves(traces)
		if curves is not None:
			info["convergence_plot"] = curves

		profiles = self._plot_profiles(traces)
		if profiles is not None:
			info["profile_plot"] = profiles

		return info
