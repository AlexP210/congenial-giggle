"""Success rate under cool-set and side-set lighting for TD-MPC2 and S2P, from run_lighting_cool_side_test.sh.

One row per task, with a cool-set and a side-set panel in each; "default" is amount 0 on both.
Plots real_task_planning episode_success_at_end_rate +/- SEM, and prints every condition with both
evaluators, marking those where TD-MPC2 has lost at least half its default success rate while S2P
has kept at least 80% of its own.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt

from plot_lighting_breakpoint import AGENTS, load_runs

RUN_NAME = re.compile(r"CoolSide_(tdmpc2|s2p)_([a-z_]+?)_(default|cool-set-[0-9.]+|side-set-[0-9.]+)\b")
SHIFTS = {"cool-set": "Cool tint (0 = 6500 K, 1 = 12000 K)", "side-set": "Key light angle (0 = default, 1 = side)"}
METRIC = "evaluation/{}/episode_success_at_end_rate"


def amount(condition, shift):
	"""The shift amount `condition` sits at, or None if it is not on `shift`'s axis."""
	if condition == "default":
		return 0.0
	return float(condition[len(shift) + 1:]) if condition.startswith(shift + "-") else None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--wandb_dir", default=os.path.join(os.environ["OUTPUT_DIR"], "s2p", "wandb"))
	parser.add_argument("--tasks", nargs="+", default=["pick_cube", "push_cube", "lift_peg"])
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "lighting_cool_side", "all_tasks"))
	args = parser.parse_args()

	runs = load_runs(args.wandb_dir, RUN_NAME)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	fig, axes = plt.subplots(len(args.tasks), len(SHIFTS), figsize=(9, 3.6 * len(args.tasks)),
		sharey=True, squeeze=False)
	for row, task in enumerate(args.tasks):
		for col, (shift, xlabel) in enumerate(SHIFTS.items()):
			ax = axes[row, col]
			for agent, (label, color) in AGENTS.items():
				points = sorted(
					(amount(c, shift), s) for (a, t, c), s in runs.items()
					if t == task and a == agent and amount(c, shift) is not None
				)
				if not points:
					continue
				xs = [x for x, _ in points]
				ys = [s[METRIC.format("real_task_planning")] for _, s in points]
				sems = [s[METRIC.format("real_task_planning") + "_sem"] for _, s in points]
				ax.errorbar(xs, ys, yerr=sems, color=color, linewidth=2, marker="o", markersize=6,
					capsize=3, elinewidth=1, label=label, markeredgecolor="white", markeredgewidth=1.5)
				ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
					va="center", fontsize=9, color="#3d3d3a")
			ax.set_title(f"{task} — {shift}", fontsize=10, color="#3d3d3a")
			ax.set_xlabel(xlabel)
			ax.set_ylim(0, 1.05)
			ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
			ax.set_axisbelow(True)
			for side in ("top", "right"):
				ax.spines[side].set_visible(False)
			ax.margins(x=0.12)
		axes[row, 0].set_ylabel("Success rate at episode end")
	axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
	fig.suptitle("Real-time planning under lighting shifts (50 episodes, ±SEM)", fontsize=11)
	fig.tight_layout()
	for ext in ("png", "pdf"):
		fig.savefig(f"{args.out}.{ext}", dpi=200)

	print(f"{'task':10s} {'condition':15s} {'TD-MPC2 real/sim':>17s} {'S2P real/sim':>13s}")
	for task in args.tasks:
		get = lambda a, c, e: runs.get((a, task, c), {}).get(METRIC.format(e))
		base = {a: get(a, "default", "real_task_planning") for a in AGENTS}
		conditions = sorted({c for (_, t, c) in runs if t == task}, key=lambda c: (c != "default", c))
		for c in conditions:
			cells = []
			for a in AGENTS:
				real, sim = get(a, c, "real_task_planning"), get(a, c, "simulation_task_planning")
				cells.append("   -/-  " if real is None else f"{real:.2f}/{sim:.2f}")
			tdmpc2, s2p = get("tdmpc2", c, "real_task_planning"), get("s2p", c, "real_task_planning")
			flag = ""
			if None not in (tdmpc2, s2p, base["tdmpc2"], base["s2p"]) and c != "default":
				if tdmpc2 <= 0.5 * base["tdmpc2"] and s2p >= 0.8 * base["s2p"]:
					flag = "  <-- TD-MPC2 fails, S2P holds"
			print(f"{task:10s} {c:15s} {cells[0]:>17s} {cells[1]:>13s}{flag}")
	print(f"Saved {args.out}.png/.pdf")


if __name__ == "__main__":
	main()
