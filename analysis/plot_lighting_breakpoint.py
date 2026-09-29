"""Success rate vs brightness for TD-MPC2 and S2P, from run_lighting_breakpoint_test.sh.

Reads each run's summary under $OUTPUT_DIR/s2p/wandb, keyed by the run_name the sweep gives it
(BrightBreakpoint_<agent>_<task>_<ambient>-<lights>); when a run_name was evaluated more than
once, the latest run wins. Plots episode_success_at_end_rate +/- SEM with one row per task and the
real-time and simulation evaluators side by side, with the lights' level (1.0 = default) on the
x-axis.
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt

AGENTS = {"tdmpc2": ("TD-MPC2", "#2a78d6"), "s2p": ("S2P", "#eb6834")}
EVALUATORS = {"real_task_planning": "Real-time planning", "simulation_task_planning": "Simulation planning"}
RUN_NAME = re.compile(r"BrightBreakpoint_(tdmpc2|s2p)_([a-z_]+?)_([0-9.]+)-([0-9.]+)")


def read_offline_run(run_dir):
	"""(run name, summary) from an offline run, which keeps both only in its binary .wandb log."""
	from wandb.proto import wandb_internal_pb2
	from wandb.sdk.internal import datastore

	(log_path,) = glob.glob(os.path.join(run_dir, "*.wandb"))
	store = datastore.DataStore()
	store.open_for_scan(log_path)
	name, summary = "", {}
	while (data := store.scan_data()) is not None:
		record = wandb_internal_pb2.Record()
		record.ParseFromString(data)
		kind = record.WhichOneof("record_type")
		if kind == "run":
			name = record.run.display_name
		elif kind == "summary":
			for item in record.summary.update:
				summary[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
	return name, summary


def load_runs(wandb_dir, pattern=RUN_NAME):
	"""{`pattern`'s groups: summary} for every finished run whose name `pattern` matches."""
	runs = {}
	# Run directories are named [offline-]run-<YYYYmmdd_HHMMSS>-<id>; sort on the timestamp so
	# online and offline runs interleave chronologically.
	run_dirs = glob.glob(os.path.join(wandb_dir, "run-*")) + glob.glob(os.path.join(wandb_dir, "offline-run-*"))
	for run_dir in sorted(run_dirs, key=lambda d: os.path.basename(d).split("run-", 1)[1]):
		try:
			if os.path.basename(run_dir).startswith("offline-"):
				name, summary = read_offline_run(run_dir)
				match = pattern.search(name)
			else:
				with open(os.path.join(run_dir, "files", "config.yaml")) as f:
					match = pattern.search(f.read())
				with open(os.path.join(run_dir, "files", "wandb-summary.json")) as f:
					summary = json.load(f)
		except (FileNotFoundError, ValueError):
			continue
		if match is None or "evaluation/real_task_planning/episode_success_at_end_rate" not in summary:
			continue
		runs[match.groups()] = summary
	return runs


def plot_panel(ax, runs, task, evaluator, table):
	for agent, (label, color) in AGENTS.items():
		points = sorted((float(lights), s) for (a, t, _, lights), s in runs.items() if t == task and a == agent)
		if not points:
			continue
		xs = [lights for lights, _ in points]
		ys = [s[f"evaluation/{evaluator}/episode_success_at_end_rate"] for _, s in points]
		sems = [s[f"evaluation/{evaluator}/episode_success_at_end_rate_sem"] for _, s in points]
		ax.errorbar(xs, ys, yerr=sems, color=color, linewidth=2, marker="o", markersize=6,
			capsize=3, elinewidth=1, label=label, markeredgecolor="white", markeredgewidth=1.5)
		ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
			va="center", fontsize=9, color="#3d3d3a")
		table += [(task, evaluator, label, x, y, e) for x, y, e in zip(xs, ys, sems)]
	ax.set_xlabel("Light level (x default; ambient scales with it)")
	ax.set_ylim(0, 1.05)
	ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.margins(x=0.12)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--wandb_dir", default=os.path.join(os.environ["OUTPUT_DIR"], "s2p", "wandb"))
	parser.add_argument("--tasks", nargs="+", default=["pick_cube"])
	parser.add_argument("--out", default=None, help="path without extension; defaults to "
		"results/lighting_breakpoint/<task> for one task and .../all_tasks for several")
	args = parser.parse_args()
	if args.out is None:
		name = args.tasks[0] if len(args.tasks) == 1 else "all_tasks"
		args.out = os.path.join(os.path.dirname(__file__), "results", "lighting_breakpoint", name)

	runs = load_runs(args.wandb_dir)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	fig, axes = plt.subplots(len(args.tasks), len(EVALUATORS), figsize=(9, 3.6 * len(args.tasks)),
		sharey=True, squeeze=False)
	table = []
	for row, task in enumerate(args.tasks):
		for col, (evaluator, title) in enumerate(EVALUATORS.items()):
			plot_panel(axes[row, col], runs, task, evaluator, table)
			axes[row, col].set_title(f"{task} — {title}", fontsize=10, color="#3d3d3a")
		axes[row, 0].set_ylabel("Success rate at episode end")
	axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
	fig.suptitle("Increasing brightness (50 episodes, ±SEM)", fontsize=11)
	fig.tight_layout()
	for ext in ("png", "pdf"):
		fig.savefig(f"{args.out}.{ext}", dpi=200)

	for row in table:
		print("{:10s} {:26s} {:8s} lights={:.4f} success={:.2f} ±{:.3f}".format(*row))
	print(f"Saved {args.out}.png/.pdf")


if __name__ == "__main__":
	main()
