import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from PIL import Image

from s2p.tasks.dmcontrol_task import DMControlTask

# ---------------------------------------------------------------------------
# Cheetah
# ---------------------------------------------------------------------------
from tdmpc2.envs.tasks.cheetah import Physics as _CheetahPhysics
from tdmpc2.envs.tasks.cheetah import get_model_and_assets as _cheetah_model_assets

_CHEETAH_PHYSICS = None


def _get_cheetah_physics():
    global _CHEETAH_PHYSICS
    if _CHEETAH_PHYSICS is None:
        xml, assets = _cheetah_model_assets()
        _CHEETAH_PHYSICS = _CheetahPhysics.from_xml_string(xml, assets)
    return _CHEETAH_PHYSICS


def obs_to_cheetah_physics(obs):
    """
    Observation layout (matches dm_control Cheetah.get_observation):
      obs[0:8]  = qpos[1:] = [rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
      obs[8:17] = qvel     = [rootx_vel, rootz_vel, rooty_vel, bthigh_vel, bshin_vel,
                               bfoot_vel, fthigh_vel, fshin_vel, ffoot_vel]

    qpos[0] (rootx) is set to 0 — not observed, rewards are translation-invariant.
    """
    obs = np.asarray(obs, dtype=np.float64)
    physics = _get_cheetah_physics()
    physics.data.qpos[0] = 0.0
    physics.data.qpos[1:] = obs[:8]
    physics.data.qvel[:] = obs[8:17]
    physics.forward()
    return physics


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------
from dm_control.suite import walker as _walker_suite
from tdmpc2.envs.tasks.walker import get_model_and_assets as _walker_model_assets

_WALKER_PHYSICS = None


def _get_walker_physics():
    global _WALKER_PHYSICS
    if _WALKER_PHYSICS is None:
        xml, assets = _walker_model_assets()
        _WALKER_PHYSICS = _walker_suite.Physics.from_xml_string(xml, assets)
    return _WALKER_PHYSICS


def obs_to_walker_physics(obs):
    """
    Observation layout (matches dm_control Walker.get_observation):
      obs[0:14]  = orientations = xmat[1:, ['xx', 'xz']].ravel()  (7 bodies × 2)
      obs[14]    = height       = xpos['torso', 'z']
      obs[15:24] = velocity     = qvel

    Body order: torso, right_thigh, right_leg, right_foot,
                left_thigh, left_leg, left_foot.

    Global body angle θ_i = atan2(xz_i, xx_i) for a y-axis rotation.
    qpos joint angles are recovered as differences along each kinematic chain.
    qpos[0] (rootx) is set to 0 — not observed, rendering is translation-invariant.
    """
    obs = np.asarray(obs, dtype=np.float64)
    orientations = obs[:14].reshape(7, 2)   # [7, (xx, xz)]
    height = float(obs[14])
    velocity = obs[15:24]

    # Global y-axis rotation angles for each body
    angles = np.arctan2(orientations[:, 1], orientations[:, 0])
    # indices: 0=torso, 1=r_thigh, 2=r_leg, 3=r_foot, 4=l_thigh, 5=l_leg, 6=l_foot

    physics = _get_walker_physics()
    physics.data.qpos[0] = 0.0          # rootx (not observed)
    physics.data.qpos[1] = height        # rootz
    physics.data.qpos[2] = angles[0]     # torso global angle
    physics.data.qpos[3] = angles[1] - angles[0]   # right_thigh relative to torso
    physics.data.qpos[4] = angles[2] - angles[1]   # right_leg relative to right_thigh
    physics.data.qpos[5] = angles[3] - angles[2]   # right_foot relative to right_leg
    physics.data.qpos[6] = angles[4] - angles[0]   # left_thigh relative to torso
    physics.data.qpos[7] = angles[5] - angles[4]   # left_leg relative to left_thigh
    physics.data.qpos[8] = angles[6] - angles[5]   # left_foot relative to left_leg
    physics.data.qvel[:] = velocity
    physics.forward()
    return physics


# ---------------------------------------------------------------------------
# Dispatch registry — add new embodiments here
# ---------------------------------------------------------------------------
_OBS_TO_PHYSICS = {
    "cheetah": obs_to_cheetah_physics,
    "walker": obs_to_walker_physics,
}


def get_obs_to_physics_fn(embodiment: str):
    if embodiment not in _OBS_TO_PHYSICS:
        raise ValueError(
            f"Unknown embodiment '{embodiment}'. Available: {list(_OBS_TO_PHYSICS)}"
        )
    return _OBS_TO_PHYSICS[embodiment]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        help="DMControl task name, e.g. cheetah-jump, walker-walk")
    parser.add_argument("--training-data-path", type=str, required=True)
    parser.add_argument("--validation-data-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str,
                        default="/path/to/datasets/tdmpc2/")
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--stack", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    W, H, S = args.width, args.height, args.stack

    embodiment = args.task.split("-")[0]
    obs_to_physics = get_obs_to_physics_fn(embodiment)

    task = DMControlTask(
        cfg=OmegaConf.create(
            {
                "task_name": args.task,
                "training_data_path": args.training_data_path,
                "validation_data_path": args.validation_data_path,
                "seed": 1,
                "horizon": None,
                "fraction": args.fraction,
                "overlap_ratio": 0.0,
                "load_device": "cpu",
                "batch_device": "cpu",
            }
        )
    )

    training_dataset, validation_dataset = task.make_dataset()

    for split_name, dataset in {
        "training": training_dataset,
        "validation": validation_dataset,
    }.items():
        # Gather selected trajectories across all files into a single dict
        all_raw = []
        for file_idx in range(len(dataset.paths)):
            raw = dataset.raw_data[file_idx]
            selected = dataset.trajectories_to_use[file_idx]
            all_raw.append({key: raw[key][selected] for key in dataset.keys})
        combined = {key: torch.cat([r[key] for r in all_raw], dim=0) for key in dataset.keys}

        N, T = combined["obs"].shape[:2]
        # uint8 keeps memory ~4x smaller than float32 for image data
        # obs shape: [N, T, 3*S, H, W] — S frames stacked along channel dim
        images = torch.zeros(N, T, 3 * S, H, W, dtype=torch.uint8)

        for traj_idx in tqdm.tqdm(range(N), desc=f"Rendering {split_name}"):
            prev_stacked = None
            for trans_idx in range(T):
                obs = combined["obs"][traj_idx, trans_idx]
                physics = obs_to_physics(obs)
                frame = torch.from_numpy(
                    physics.render(width=W, height=H, camera_id=args.camera_id).copy()
                ).permute(2, 0, 1)  # [3, H, W]

                if traj_idx == 0 and trans_idx == 0 and split_name == "training":
                    png_path = f"{args.data_dir}/sanity_check.png"
                    Image.fromarray(frame.permute(1, 2, 0).numpy()).save(png_path)
                    print(f"Sanity check image saved to {png_path}")

                if trans_idx == 0:
                    stacked = frame.repeat(S, 1, 1)  # [3*S, H, W]
                else:
                    stacked = torch.cat([prev_stacked[3:], frame], dim=0)

                images[traj_idx, trans_idx] = stacked
                prev_stacked = stacked

        combined["obs"] = images

        out_path = f"{args.data_dir}/mt30-visual-{args.task}-{split_name}.pt"
        torch.save(combined, out_path)
        print(f"Saved {split_name} dataset to {out_path}")
