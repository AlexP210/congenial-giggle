import tensordict
import torch
import numpy as np

def diag_gaussian_log_prob(x, mean, std):
    var = std**2
    return -0.5 * (((x-mean)**2 / var) + torch.log(var) + np.log(2*torch.pi)).sum(-1)

def squeeze_collate(batch):
    return batch[0]

def squashing_fn(x, a, b):
    """Compute bounded reward function."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

def collate_batch(batch):
    """Pass through a batch the dataset already stacked; stack it here if it didn't.

    `TrajectoryWindowDataset.__getitems__` returns a whole batch as one TensorDict, and
    torch's fetcher still hands whatever it got to `collate_fn`. A dataset without that
    hook (or an older torch that ignores it) yields the per-sample list instead, which
    still needs stacking -- so every DataLoader over these datasets can share one
    collate_fn regardless of which path its dataset takes.
    """
    return tensordict.stack(batch) if isinstance(batch, list) else batch


def infinite_loader(loader:torch.utils.data.DataLoader):
    while True:
        for batch in loader:
            yield batch

def episode_boundary(terminated, truncated) -> bool:
	"""
	Whether the episode is over, for a batch of envs stepping in lockstep.

	Envs collected or evaluated in parallel are reset together, so they have to finish together:
	`custom_maniskill_tasks.FrameStack` keeps one frame buffer for the whole batch and refuses a
	partial reset, and a replay window that spanned one env's reset would decode to frames the env
	never produced. That holds for the tasks here -- the `-v1.1` ids suppress termination in the
	task itself, leaving the shared time limit as the only done signal -- but it is a property of
	the task rather than of any one loop, so it is checked rather than assumed. A task that does
	end an episode early is told so here, instead of silently poisoning a buffer or scoring
	episodes of different lengths against each other.
	"""
	done = torch.as_tensor(terminated) | torch.as_tensor(truncated)
	if bool(done.any()) != bool(done.all()):
		raise ValueError(
			f"{int(done.sum())} of {done.numel()} envs finished their episode while the rest are "
			"still running. Parallel envs are stepped in lockstep and reset together, which needs "
			"every episode to end on the same step; this task ends them early or at staggered "
			"time limits. Run it with num_envs=1."
		)
	return bool(done.all())


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	"""Sample an index from the categorical `p`, along `dim`. Implementation from TD-MPC2.

	Returns the argmax over `dim`, so the answer has `p`'s shape with `dim` removed: a 1-D `p`
	gives a scalar, and a `[num_envs, num_categories]` one sampled along `dim=1` gives one index
	per env.

	`dim` is honoured rather than assumed to be the last axis. `softmax` is monotone within a
	slice, so an argmax hard-coded to `-1` happens to agree with any `dim` *when `dim` is already
	the last one* -- and silently samples along the wrong axis when it is not.
	"""
	logits = p.log()
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(dim)
