"""One env, presented as a batch of one.

Every online env in this project is batched -- observations, `reward`, `terminated` and
`truncated` carry a leading env axis, and `step` takes `(num_envs, *action_dimension)`. ManiSkill
is batched natively; dm_control is not, and cannot be: it steps a single simulator.
This wrapper closes that gap so the collection and evaluation loops never have to ask which kind
of task they are driving.

It sits *outermost*, above `FrameStack`, so the adapters underneath keep serving exactly what they
serve today (`frame_axis=0`, unbatched frames) and none of them are modified. What comes out is
`(1, num_frames, *feature_dims)` -- the same layout `ManiSkillTask` produces at `num_envs=1`.
"""

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase


def _batch(value):
    """`value` with a leading axis of size 1, over a dict/TensorDict tree."""
    if isinstance(value, TensorDictBase):
        return TensorDict(
            {key: _batch(item) for key, item in value.items()}, batch_size=[1]
        )
    if isinstance(value, dict):
        return {key: _batch(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return value[None]
    return value


def _flag(value):
    """A `terminated`/`truncated` flag as a `(1,)` bool tensor."""
    return torch.as_tensor(value, dtype=torch.bool).reshape(1)


class SingleEnvBatch(gym.Wrapper):
    """Present a single env as a batch of one; see the module docstring.

    `reset` mirrors whatever the wrapped env returns -- the observation alone (which is what the
    S2P observation adapters and `FrameStack` hand back) or an `(observation, info)` pair -- so
    this can be dropped on top of either without the caller noticing.

    `info` is passed through untouched. Its entries are heterogeneous (per-env flags, but also
    strings and nested dicts), so batching it would mean guessing which leaves are per-env;
    callers that read a per-env flag out of it normalize at the point of use instead.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_envs = 1
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.observation_space = gym.vector.utils.batch_space(env.observation_space, n=1)
        self.action_space = gym.vector.utils.batch_space(env.action_space, n=1)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
            return _batch(observation), info
        return _batch(result)

    def _unbatch_action(self, action):
        """`action` without its env axis, for the unbatched env underneath.

        The axis is required rather than tolerated. An unbatched `(A,)` action indexed as
        `action[0]` would come out a scalar and be applied as one, which for a Panda's 4-D pose
        or a cheetah's 6-D torque is a silent, plausible-looking wrong move rather than an
        error -- so the rank is checked instead of assumed.
        """
        expected = (1, *self.single_action_space.shape)
        if tuple(getattr(action, "shape", ())) != expected:
            raise ValueError(
                f"{type(self).__name__} takes a batched action of shape {expected}, got "
                f"{tuple(getattr(action, 'shape', ()))}. Every env in this project is batched; "
                "add the env axis rather than passing a single env's action."
            )
        return action[0]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            self._unbatch_action(action)
        )
        return (
            _batch(observation),
            torch.as_tensor(reward, dtype=torch.float32).reshape(1),
            _flag(terminated),
            _flag(truncated),
            info,
        )

    def rand_act(self):
        """A random action in the batched action space, as `(1, *action_dimension)`.

        Taken from the wrapped env's own `rand_act` where it has one, so that an adapter which
        samples something other than a uniform draw over its Box (or which returns a dtype of its
        own) still decides what a random action is.
        """
        try:
            action = self.env.get_wrapper_attr("rand_act")()
        except (AttributeError, TypeError):
            action = torch.from_numpy(
                np.asarray(self.single_action_space.sample(), dtype=np.float32)
            )
        return action.unsqueeze(0)

    def __getattr__(self, name):
        """Forward unknown public attributes inwards.

        gymnasium >= 1.0 dropped `Wrapper.__getattr__`, so without this a caller reaching through
        for the wrapped env's own API sees only this wrapper.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        try:
            return env.get_wrapper_attr(name)
        except (AttributeError, TypeError):
            raise AttributeError(
                f"{type(self).__name__} and the envs it wraps have no attribute {name!r}"
            ) from None
