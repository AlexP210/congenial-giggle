"""Dataset frameskip semantics against a synthetic h5 with hand-checkable values.

Run with `python agents/squeeze2plan/tests/test_frameskip.py` (no pytest in the project environment).

Alignment convention (set by `_pad_transitions`, and matching the online trainer): every
transition-indexed leaf is shifted by one row, so index i holds the action/reward/flag that
brought the episode *to* obs[i], and index 0 is a NaN pad. Primitive action i-1 == padded row i.

Synthetic episode: obs[i] = i, padded action[i] = [i-1, i-0.5], padded reward[i] = i-1,
state puck[i] = 10i, and the final step truncates.
"""
import os
import tempfile

import h5py
import numpy as np
import torch
from mani_skill.utils.io_utils import dump_json

from s2p.lib.maniskill_transition_dataset import ManiSkillTrajectoryDataset

PATH = os.path.join(tempfile.gettempdir(), "s2p_frameskip_fixture")
T = 12  # primitive steps


def build():
    with h5py.File(f"{PATH}.h5", "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("obs", data=np.arange(T + 1, dtype=np.float32)[:, None])
        g.create_dataset("actions", data=np.stack(
            [np.arange(T, dtype=np.float32), np.arange(T, dtype=np.float32) + 0.5], -1))
        g.create_dataset("rewards", data=np.arange(T, dtype=np.float32))
        g.create_dataset("terminated", data=np.zeros(T, dtype=bool))
        trunc = np.zeros(T, dtype=bool); trunc[-1] = True
        g.create_dataset("truncated", data=trunc)
        g.create_group("env_states").create_dataset(
            "puck", data=np.arange(T + 1, dtype=np.float32)[:, None] * 10)
    dump_json(f"{PATH}.json", {
        "env_info": {"env_id": "Fake-v1", "env_kwargs": {}, "max_episode_steps": T},
        "episodes": [{"episode_id": 0, "elapsed_steps": T, "success": True,
                      "control_mode": "pd_ee_delta_pos", "episode_seed": 0, "reset_kwargs": {}}],
    })


STRUCTURE = {
    "obs": {"x": "obs"},
    "action": "actions",
    "reward": "rewards",
    "terminated": "terminated",
    "truncated": "truncated",
    "state": {"puck": "env_states/puck"},
}


def load(horizon, num_frames, frame_skip):
    return ManiSkillTrajectoryDataset(
        dataset_file=f"{PATH}.h5", json_file=f"{PATH}.json", structure=STRUCTURE,
        horizon=horizon, num_frames=num_frames, frame_skip=frame_skip, load=True,
    )


def test_frame_skip_one_is_unchanged():
    """frame_skip=1 must reproduce the pre-frameskip behaviour exactly, NaN pad included."""
    ds = load(horizon=3, num_frames=2, frame_skip=1)
    s = ds[0]
    assert s["obs"]["x"].squeeze(-1).tolist() == [[0, 0], [0, 1], [1, 2]], s["obs"]["x"]
    a = s["action"].tolist()
    assert np.isnan(a[0]).all() and a[1:] == [[0, 0.5], [1, 1.5]], a
    r = s["reward"].squeeze(-1).tolist()
    assert np.isnan(r[0]) and r[1:] == [0, 1], r
    assert s["state"]["puck"].squeeze(-1).tolist() == [0, 10, 20]
    assert [t for _, t in ds.valid_indices] == [0, 3, 6, 9], ds.valid_indices


def test_macro_step_aggregation():
    k, H, n = 3, 3, 2
    s = load(horizon=H, num_frames=n, frame_skip=k)[0]
    # boundaries at primitive 0, 3, 6; the stack strides by k and clamps at the episode start
    assert s["obs"]["x"].squeeze(-1).tolist() == [[0, 0], [0, 3], [3, 6]], s["obs"]["x"]
    # one macro action == the k primitive actions that brought us to the boundary, concatenated
    assert s["action"].shape == (H, k * 2), s["action"].shape
    assert np.isnan(s["action"][0].tolist()).all(), "step 0 has no incoming actions -> NaN pad"
    assert s["action"][1].tolist() == [0, 0.5, 1, 1.5, 2, 2.5], s["action"][1]
    assert s["action"][2].tolist() == [3, 3.5, 4, 4.5, 5, 5.5], s["action"][2]
    # rewards sum over the same chunk
    r = s["reward"].squeeze(-1).tolist()
    assert np.isnan(r[0]) and r[1:] == [0 + 1 + 2, 3 + 4 + 5], r
    # state is read at the boundary, never aggregated
    assert s["state"]["puck"].squeeze(-1).tolist() == [0, 30, 60]
    # flags OR over the chunk (NaN pad preserved on step 0, as in the unskipped path)
    term = s["terminated"].squeeze(-1).tolist()
    assert np.isnan(term[0]) and term[1:] == [0, 0], term


def test_truncation_flag_is_or_ed_over_the_chunk():
    """The chunk containing the episode's final primitive step must report truncation."""
    s = load(horizon=5, num_frames=1, frame_skip=3)[0]   # boundaries 0,3,6,9,12
    trunc = s["truncated"].squeeze(-1).tolist()
    assert np.isnan(trunc[0]) and trunc[1:] == [0, 0, 0, 1], trunc


def test_window_count_scales_with_frame_skip():
    # the last macro step of a window at t sits at t + (H-1)*k, which must stay <= T
    assert [t for _, t in load(2, 1, 3).valid_indices] == [0, 6], load(2, 1, 3).valid_indices
    assert [t for _, t in load(5, 1, 3).valid_indices] == [0], load(5, 1, 3).valid_indices
    assert len(load(6, 1, 3)) == 0, "a window longer than the episode yields nothing"


def test_stacked_observation_strides_by_frame_skip():
    obs = load(horizon=2, num_frames=3, frame_skip=3).stacked_observation(0, step=-1)
    assert obs["x"].squeeze(-1).tolist() == [6, 9, 12], obs["x"]


def test_matches_the_online_env():
    """The real check: a window's macro actions, replayed in the env built for the same skip."""
    from custom_maniskill_tasks import make_env
    k, H = 3, 4
    env = make_env("PushCube-v1", obs_mode="state", frame_skip=k, sim_backend="physx_cpu")
    env.reset(seed=17)
    actions = [torch.rand(k * 4) * 0.2 for _ in range(H)]
    macro_obs, macro_rew = [], []
    for a in actions:
        o, r, _, _, _ = env.step(a)
        macro_obs.append(o.clone()); macro_rew.append(float(r.reshape(-1)[0]))
    env.close()

    plain = make_env("PushCube-v1", obs_mode="state", sim_backend="physx_cpu")
    plain.reset(seed=17)
    prim_obs, prim_rew = [], []
    for a in actions:
        for i in range(k):
            o, r, _, _, _ = plain.step(a[i * 4:(i + 1) * 4])
            prim_obs.append(o.clone()); prim_rew.append(float(r.reshape(-1)[0]))
    plain.close()

    for j in range(H):
        chunk = prim_rew[j * k:(j + 1) * k]
        assert abs(macro_rew[j] - sum(chunk)) < 1e-5, (j, macro_rew[j], sum(chunk))
        # the macro observation is the LAST primitive obs of the chunk, not a gather of them
        assert torch.allclose(macro_obs[j], prim_obs[(j + 1) * k - 1]), j
    print(f"      (macro reward {macro_rew[0]:.4f} == sum {sum(prim_rew[:k]):.4f}; "
          f"obs == last primitive step of each chunk)")


if __name__ == "__main__":
    build()
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
