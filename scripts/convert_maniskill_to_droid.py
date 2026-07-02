"""
Convert ManiSkill trajectory .h5 (with RGB obs) to EXPO-FT's DROID-style format.
Config-driven: reads control mode, state obs key, action dims from task YAML.

Output structure:
    output_dir/
        0/traj.hdf5
        1/traj.hdf5
        ...

Each traj.hdf5 contains:
    saved_observation/
        observation/exterior_image_1_left  (T, H, W, 3) uint8
        observation/wrist_image_left       (T, H, W, 3) uint8
        observation/joint_position         (T, 7)  float32  [j0..j6]   (joint mode)
        observation/cartesian_position     (T, 6)  float32  [x,y,z,r,p,y] (cartesian mode)
        observation/gripper_position       (T, 1)  float32
        prompt                             (T,)    str
    action/
        joint_position                     (T, 7)  float32  (joint mode)
        cartesian_velocity                 (T, 6)  float32  (cartesian mode)
        gripper_position OR gripper_velocity (T, 1) float32
"""

import argparse
import os
import h5py
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def convert(traj_path: str, output_dir: str, task_description: str, cfg=None, max_episodes: int = None):
    os.makedirs(output_dir, exist_ok=True)

    # Config-driven dimensions
    state_obs_key  = getattr(cfg, 'state_obs_key',     'observation/cartesian_position')
    state_obs_dim  = getattr(cfg, 'state_obs_dim',     6)
    action_dim     = getattr(cfg, 'output_action_dim', 7)
    action_space   = getattr(cfg, 'action_space',      'cartesian_velocity')
    gripper_space  = getattr(cfg, 'gripper_action_space', 'velocity')
    arm_action_dim = action_dim - 1  # everything except gripper

    with h5py.File(traj_path, "r") as f:
        episode_keys = list(f.keys())
        if max_episodes is not None:
            episode_keys = episode_keys[:max_episodes]

        print(f"Converting {len(episode_keys)} episodes...")
        print(f"  state_obs_key={state_obs_key}, state_obs_dim={state_obs_dim}")
        print(f"  action_dim={action_dim} (arm={arm_action_dim}, gripper=1)")
        print(f"  action keys: {action_space}, {gripper_space}")

        for ep_idx, ep_key in enumerate(tqdm(episode_keys)):
            ep = f[ep_key]

            rgb_base  = np.array(ep["obs/sensor_data/base_camera/rgb"])   # (T, H, W, 3)
            if "hand_camera" in ep["obs/sensor_data"]:
                rgb_wrist = np.array(ep["obs/sensor_data/hand_camera/rgb"])   # (T, H, W, 3)
            else:
                # No wrist camera available for this task (e.g. PushCube) — use a black image.
                rgb_wrist = np.zeros_like(rgb_base)
                rgb_wrist[:, 0, 0] = 2
            tcp_pose  = np.array(ep["obs/extra/tcp_pose"])                 # (T, 7)
            qpos      = np.array(ep["obs/agent/qpos"])                     # (T, 9)
            actions   = np.array(ep["actions"])                            # (T-1, action_dim)

            T = rgb_base.shape[0]

            # State observation
            if state_obs_key == 'observation/joint_position':
                state = qpos[:, :state_obs_dim].astype(np.float32)        # (T, 7)
            else:
                xyz       = tcp_pose[:, :3]
                quat_xyzw = tcp_pose[:, 3:]
                euler     = np.stack([quat_to_euler(q) for q in quat_xyzw])
                state     = np.concatenate([xyz, euler], axis=-1).astype(np.float32)  # (T, 6)

            # Gripper position: first finger (both fingers always identical on Panda)
            gripper = qpos[:, 7:8].astype(np.float32)                     # (T, 1)

            # Actions: arm + gripper, pad last timestep with zeros
            act_arm  = actions[:, :arm_action_dim].astype(np.float32)
            act_grip = actions[:, arm_action_dim:arm_action_dim+1].astype(np.float32)
            act_arm  = np.concatenate([act_arm,  np.zeros((1, arm_action_dim), dtype=np.float32)], axis=0)
            act_grip = np.concatenate([act_grip, np.zeros((1, 1),             dtype=np.float32)], axis=0)

            prompts = np.array([task_description] * T)

            ep_dir = os.path.join(output_dir, str(ep_idx))
            os.makedirs(ep_dir, exist_ok=True)

            with h5py.File(os.path.join(ep_dir, "traj.hdf5"), "w") as out:
                obs = out.create_group("saved_observation")
                obs.create_dataset("observation/exterior_image_1_left", data=rgb_base,  dtype=np.uint8)
                obs.create_dataset("observation/wrist_image_left",       data=rgb_wrist, dtype=np.uint8)
                obs.create_dataset(state_obs_key,                        data=state)
                obs.create_dataset("observation/gripper_position",       data=gripper)
                dt = h5py.string_dtype()
                obs.create_dataset("prompt", data=prompts.astype(bytes), dtype=dt)

                act = out.create_group("action")
                act.create_dataset(action_space,  data=act_arm)
                act.create_dataset(gripper_space, data=act_grip)

    print(f"Done. {len(episode_keys)} episodes written to {output_dir}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from expo_ft.utils.config_loader import load_task_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path",        required=True)
    parser.add_argument("--output-dir",       required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--config",           default="configs/task/maniskill_stack_cube.yaml")
    parser.add_argument("--max-episodes",     type=int, default=None)
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    convert(
        traj_path=args.traj_path,
        output_dir=args.output_dir,
        task_description=args.task_description,
        cfg=cfg,
        max_episodes=args.max_episodes,
    )
