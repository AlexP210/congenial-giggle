import numpy as np
import matplotlib.pyplot as plt

# student_y = np.load("student_cheetah_flip_return_means.npy").flatten()
# student_err = np.load("student_cheetah_flip_return_sem.npy").flatten()
# student_x = np.load("student_cheetah_flip_plan_latency_means.npy").flatten()

# teacher_y = np.load("teacher_cheetah_flip_finetuned_return_means.npy").flatten()
# teacher_err = np.load("teacher_cheetah_flip_finetuned_return_sem.npy").flatten()
# teacher_x = np.load("teacher_cheetah_flip_finetuned_plan_latency_means.npy").flatten()

# expert_y = np.load("expert_cheetah_flip_return_means.npy").flatten()
# expert_err = np.load("expert_cheetah_flip_return_sem.npy").flatten()
# expert_x = np.load("expert_cheetah_flip_plan_latency_means.npy").flatten()

repo_expert_noteacher_y = np.load("repo_cheetah_jump_offline_noteacher_return_means.npy").flatten()
repo_expert_noteacher_err = np.load("repo_cheetah_jump_offline_noteacher_return_sem.npy").flatten()
repo_expert_noteacher_x = np.load("repo_cheetah_jump_offline_noteacher_plan_latency_means.npy").flatten()

repo_expert_teacher_y = np.load("repo_cheetah_jump_offline_teacher_return_means.npy").flatten()
repo_expert_teacher_err = np.load("repo_cheetah_jump_offline_teacher_return_sem.npy").flatten()
repo_expert_teacher_x = np.load("repo_cheetah_jump_offline_teacher_plan_latency_means.npy").flatten()


def bin_max(x, y, err, bins):
    """
    For each x-bin, select the point with maximum y, then take the cumulative
    maximum so each bin reflects the best performance at that latency or lower.
    Returns:
        bin_centers
        cumulative_max_y
        corresponding_err (from whichever bin holds the running best)
    """
    inds = np.digitize(x, bins)

    bx = []
    by = []
    berr = []

    for i in range(1, len(bins)):
        mask = inds == i

        if not np.any(mask):
            continue

        x_bin = x[mask]
        y_bin = y[mask]
        err_bin = err[mask]

        max_idx = np.argmax(y_bin)

        bx.append(np.mean([bins[i - 1], bins[i]]))
        by.append(y_bin[max_idx])
        berr.append(err_bin[max_idx])

    bx = np.array(bx)
    by = np.array(by)
    berr = np.array(berr)

    # For each bin, propagate the best seen so far (cumulative max).
    # The error bar follows from the bin that holds the running best.
    running_best = 0
    for i in range(len(by)):
        if by[i] >= by[running_best]:
            running_best = i
        by[i] = by[running_best]
        berr[i] = berr[running_best]

    return bx, by, berr


# Shared bins across both datasets
all_x = np.concatenate([repo_expert_noteacher_x, repo_expert_teacher_x])

num_bins = 10
bins = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), num_bins + 1)

# Compute binned maxima

repo_expert_noteacher_bx, repo_expert_noteacher_by, repo_expert_noteacher_berr = bin_max(
    repo_expert_noteacher_x,
    repo_expert_noteacher_y,
    repo_expert_noteacher_err,
    bins,
)

repo_expert_teacher_bx, repo_expert_teacher_by, repo_expert_teacher_berr = bin_max(
    repo_expert_teacher_x,
    repo_expert_teacher_y,
    repo_expert_teacher_err,
    bins,
)


# Plot
plt.figure(figsize=(8, 5))

# plt.errorbar(
#     student_bx,
#     student_by,
#     yerr=student_berr,
#     fmt='o-',
#     capsize=4,
#     label='Student',
# )

# plt.errorbar(
#     teacher_bx,
#     teacher_by,
#     yerr=teacher_berr,
#     fmt='s-',
#     capsize=4,
#     label='TD-MPC2 Multi-Task Teacher (Fine-Tuned)',
# )

# plt.errorbar(
#     expert_bx,
#     expert_by,
#     yerr=expert_berr,
#     fmt='s-',
#     capsize=4,
#     label='Single-Task TD-MPC2',
# )

plt.errorbar(
    repo_expert_teacher_bx,
    repo_expert_teacher_by,
    yerr=repo_expert_teacher_berr,
    fmt='s-',
    capsize=4,
    label='Teacher+RePo',
)
plt.errorbar(
    repo_expert_noteacher_bx,
    repo_expert_noteacher_by,
    yerr=repo_expert_noteacher_berr,
    fmt='s-',
    capsize=4,
    label='RePo',
)

plt.xlabel("Planning Budget (s)", size=15)
plt.ylabel("Maximum Return in Bin", size=15)
plt.title("Episode Returns on Real-Time Cheetah-Jump", fontsize=15)
plt.legend(fontsize=15)
plt.xscale("log")
plt.grid(True)
plt.ylim(bottom=0)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.savefig("TeacherVsStudentCheetahFlip.png")