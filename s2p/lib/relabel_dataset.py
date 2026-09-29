import numpy as np
from tdmpc2.envs.tasks.cheetah import Physics, get_model_and_assets, CustomCheetah
from dm_control.suite import cheetah

from s2p.tasks.dmcontrol_task import DMControlTask
from omegaconf import OmegaConf

import torch
import tqdm

# Cache XML/assets so they are only read once per process.
_XML_STRING, _ASSETS = get_model_and_assets()


_PHYSICS = Physics.from_xml_string(_XML_STRING, _ASSETS)


def obs_to_cheetah_physics(obs):
    """
    Load obs into the shared Physics instance and run forward().

    Observation layout (matches dm_control Cheetah.get_observation):
      obs[0:8]  = qpos[1:] = [rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
      obs[8:17] = qvel     = [rootx_vel, rootz_vel, rooty_vel, bthigh_vel, bshin_vel,
                               bfoot_vel, fthigh_vel, fshin_vel, ffoot_vel]

    qpos[0] (rootx) is set to 0 — not observed, rewards are translation-invariant.
    """
    obs = np.asarray(obs, dtype=np.float64)
    _PHYSICS.data.qpos[0] = 0.0
    _PHYSICS.data.qpos[1:] = obs[:8]
    _PHYSICS.data.qvel[:] = obs[8:17]
    _PHYSICS.forward()
    return _PHYSICS

if __name__ == "__main__":
    task = DMControlTask(
        cfg=OmegaConf.create(
            {
              "task_name": "cheetah-jump",
              "data_root": "/path/to/tdmpc2_data/datasets/mt30/",
              "seed": 1,
              "horizon": None,
              "fraction": 1.0,
              "overlap_ratio": 0.0,
              "load_device": "cpu",
              "batch_device": "cpu"
            }
      )
    )

    custom_cheetah = CustomCheetah(goal='flip', move_speed=cheetah._RUN_SPEED, random=None)
    training_dataset, validation_dataset = task.make_dataset()
    c = 0
    for name, dataset in {"training": training_dataset, "validation": validation_dataset}.items():
      for trajectory_idx, trajectory in tqdm.tqdm(list(enumerate(dataset))):
          for transition_idx, transition in list(enumerate(trajectory))[1:]:
            obs = transition["obs"]
            physics = obs_to_cheetah_physics(obs)
            reward = custom_cheetah._flip_reward(physics)
            dataset.data["reward"][trajectory_idx, transition_idx] = torch.Tensor([reward])
      dataset.save(f"/path/to/tdmpc2_data/datasets/mt30/mt30-cheetah-flip-{name}.pt")