"""The batched-env contract, without a simulator: lockstep flags, per-env windows, per-env plans.

Run with `python agents/squeeze2plan/tests/test_parallel_envs.py` (no pytest in the project environment).

Everything here is CPU-only and model-free. What it cannot cover is ManiSkill itself -- that the
observations really do come out `(num_envs, num_frames, *feature_dims)` on physx_cuda -- which is
checked by running the thing.
"""
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from s2p.lib.transition_data import OnlineTransitionDataset
from s2p.lib.utils import episode_boundary
from s2p.planners.mppi_planner import MPPIPlanner
from s2p.tasks.base.single_env_batch import SingleEnvBatch


# --------------------------------------------------------------------------------------------- #
# lockstep episodes
# --------------------------------------------------------------------------------------------- #


def test_episode_boundary_reports_the_shared_end():
    flags = lambda *values: torch.tensor(values, dtype=torch.bool)
    assert episode_boundary(flags(False, False), flags(False, False)) is False
    assert episode_boundary(flags(False, False), flags(True, True)) is True
    assert episode_boundary(flags(True, True), flags(False, False)) is True
    # Either flag ending the episode counts, as long as every env ends together.
    assert episode_boundary(flags(True, False), flags(False, True)) is True


def test_episode_boundary_refuses_a_staggered_end():
    """One env finishing early would put a reset inside another env's window."""
    try:
        episode_boundary(
            torch.tensor([False, False]), torch.tensor([True, False])
        )
    except ValueError as error:
        assert "1 of 2 envs" in str(error), error
        return
    raise AssertionError("expected a ValueError for a staggered episode end")


# --------------------------------------------------------------------------------------------- #
# one buffer entry per env
# --------------------------------------------------------------------------------------------- #


def _window(horizon, num_envs):
    """A `[horizon, num_envs]` window whose values identify their (step, env, key)."""
    step = torch.arange(horizon, dtype=torch.float32)[:, None]
    env = torch.arange(num_envs, dtype=torch.float32)[None, :]
    tag = step * 100 + env
    return TensorDict(
        {
            "obs": TensorDict(
                {"rgb": tag[..., None, None].expand(horizon, num_envs, 2, 2).clone()},
                batch_size=[horizon, num_envs],
            ),
            "action": torch.stack([tag, -tag], dim=-1),
            "reward": tag[..., None].clone(),
            "terminated": torch.zeros(horizon, num_envs, 1, dtype=torch.bool),
            "truncated": torch.zeros(horizon, num_envs, 1, dtype=torch.bool),
        },
        batch_size=[horizon, num_envs],
    )


def test_add_subtrajectories_stores_one_window_per_env_in_order():
    horizon, num_envs = 4, 3
    window = _window(horizon, num_envs)

    dataset = OnlineTransitionDataset(capacity=16, horizon=horizon)
    dataset.add_subtrajectories(window)
    assert len(dataset) == num_envs, len(dataset)

    for env_index in range(num_envs):
        entry = dataset[env_index]
        assert entry["action"].shape == (horizon, 2), entry["action"].shape
        assert torch.equal(entry["action"], window["action"][:, env_index]), env_index
        assert torch.equal(entry["obs"]["rgb"], window["obs"]["rgb"][:, env_index]), env_index


def test_add_subtrajectories_matches_one_at_a_time():
    """Widening the collector changes nothing about what a window in the buffer is."""
    horizon, num_envs = 4, 3
    window = _window(horizon, num_envs)

    batched = OnlineTransitionDataset(capacity=16, horizon=horizon)
    batched.add_subtrajectories(window)

    one_by_one = OnlineTransitionDataset(capacity=16, horizon=horizon)
    for env_index in range(num_envs):
        one_by_one.add_subtrajectory(window[:, env_index])

    assert len(batched) == len(one_by_one)
    for index in range(len(batched)):
        assert (batched[index] == one_by_one[index]).all(), index


# --------------------------------------------------------------------------------------------- #
# one plan per env, from that env's own state
# --------------------------------------------------------------------------------------------- #


LATENT, ACTION = 3, 2


class _Dynamics:
    """z' = z + [a, 0...]; the last latent channel (the env's target) is left alone."""

    def dynamics(self, s, a):
        pad = torch.zeros(*a.shape[:-1], LATENT - ACTION)
        return s + torch.cat([a, pad], dim=-1)


class _Reward:
    """Reward peaks where the action matches the target carried in `z[..., -1]`."""

    def reward(self, s, a):
        return -((a - s[..., -1:]) ** 2).mean(dim=-1, keepdim=True)


class _Task:
    action_dimension = (ACTION,)
    # What `MPPIPlanner` clamps its proposals to. The real tasks read this off their env's
    # action space; these tests plan in the [-1, 1] box every wrapped env here uses.
    action_limits = (
        np.full(ACTION, -1.0, dtype=np.float32),
        np.full(ACTION, 1.0, dtype=np.float32),
    )


def _planner():
    cfg = OmegaConf.create({
        "horizon": 3, "num_samples": 128, "num_elites": 16, "iterations": 6,
        "fraction_of_policy_trajectories": 0.0, "temperature": 0.5,
        "min_std": 0.05, "max_std": 2.0, "discount": 0.99, "use_value": False,
        "device": "cpu",
    })
    return MPPIPlanner(cfg=cfg, task=_Task())


def _plan(planner, state, prior, eval_mode=True, seed=7):
    torch.manual_seed(seed)
    # `plan` returns (actions, info); these tests are about the actions. The info dict carries
    # the planner's convergence trace, which `PlannerConvergenceEvaluator` is what reads.
    plan, _ = planner.plan(
        dynamics_model=_Dynamics(), reward_model=_Reward(), value_model=None,
        policy_model=None, current_state=state, eval_mode=eval_mode, action_prior=prior,
    )
    return plan


def test_each_env_plans_against_its_own_state():
    """The failure this guards is silent: a sample-major population has the same shape."""
    planner = _planner()
    targets = torch.tensor([-0.8, 0.0, 0.5, 0.9])
    z = torch.zeros(len(targets), LATENT)
    z[:, -1] = targets

    plan = _plan(
        planner,
        z.unsqueeze(0),
        torch.zeros(planner.cfg.horizon, len(targets), ACTION),
    )
    assert plan.shape == (planner.cfg.horizon, len(targets), ACTION), plan.shape

    # Every env's first action sits at its own target, not at the population's average one.
    error = (plan[0] - targets[:, None]).abs().max()
    assert error < 0.2, f"envs are planning against each other's states ({float(error):.3f})"


def test_an_unbatched_prior_gives_back_an_unbatched_plan():
    """One env with a `[T, A]` prior is exactly what it was before the env axis existed."""
    planner = _planner()
    z = torch.zeros(1, LATENT)
    z[:, -1] = 0.4

    unbatched = _plan(planner, z.unsqueeze(0), torch.zeros(planner.cfg.horizon, ACTION))
    batched = _plan(planner, z.unsqueeze(0), torch.zeros(planner.cfg.horizon, 1, ACTION))

    assert unbatched.shape == (planner.cfg.horizon, ACTION), unbatched.shape
    assert torch.equal(unbatched.unsqueeze(1), batched)


def test_a_one_env_prior_for_many_envs_is_refused():
    """Broadcasting it would plan for four envs and answer about one."""
    planner = _planner()
    try:
        _plan(planner, torch.zeros(4, LATENT).unsqueeze(0), torch.zeros(planner.cfg.horizon, ACTION))
    except ValueError as error:
        assert "one env's prior" in str(error), error
        return
    raise AssertionError("expected a ValueError for a prior that names no envs")


# --------------------------------------------------------------------------------------------- #
# a single env, wearing the batch axis
# --------------------------------------------------------------------------------------------- #


class _OneEnv(gym.Env):
    """A minimal unbatched env in the layout the S2P observation adapters produce."""

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 2), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION,), dtype=np.float32)

    def reset(self, **kwargs):
        self.last_action = None
        return torch.zeros(3, 2)

    def step(self, action):
        assert action.shape == (ACTION,), f"inner env got a batched action: {tuple(action.shape)}"
        self.last_action = action
        return torch.ones(3, 2), 1.5, False, True, {"success": True}


def test_single_env_batch_adds_and_strips_the_env_axis():
    env = SingleEnvBatch(_OneEnv())
    assert env.num_envs == 1

    obs = env.reset()
    assert obs.shape == (1, 3, 2), obs.shape

    action = env.rand_act()
    assert action.shape == (1, ACTION), action.shape

    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs.shape == (1, 3, 2), obs.shape
    assert reward.shape == (1,) and truncated.shape == (1,), (reward.shape, truncated.shape)
    assert episode_boundary(terminated, truncated) is True
    # The inner env saw its own unbatched action, and the env axis was really removed.
    assert torch.equal(env.env.last_action, action[0])


def test_single_env_batch_refuses_an_unbatched_action():
    """`action[0]` of a `(A,)` action is a scalar, which a sim would happily apply."""
    env = SingleEnvBatch(_OneEnv())
    env.reset()
    try:
        env.step(torch.zeros(ACTION))
    except ValueError as error:
        assert "batched action" in str(error), error
        return
    raise AssertionError("expected a ValueError for an action with no env axis")


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
