"""
Convert ManiSkill trajectory .h5 (with RGB obs) to EXPO-FT's DROID-style format.

Output structure:
    output_dir/
        0/traj.hdf5
        1/traj.hdf5
        ...

Each traj.hdf5 contains:
    saved_observation/
        observation/exterior_image_1_left  (T, H, W, 3) uint8
        observation/wrist_image_left       (T, H, W, 3) uint8
        observation/cartesian_position     (T, 6)  float32  [x,y,z,roll,pitch,yaw]
        observation/gripper_position       (T, 1)  float32
        prompt                             (T,)    str
    action/
        cartesian_velocity                 (T, 6)  float32
        gripper_velocity                   (T, 1)  float32
"""

import argparse
import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to euler angles (roll,pitch,yaw) in radians."""
    r = Rotation.from_quat(quat_xyzw)  # scipy expects (x,y,z,w)
    return r.as_euler("xyz", degrees=False)


def convert(traj_path: str, output_dir: str, task_description: str, max_episodes: int = None):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(traj_path, "r") as f:
        episode_keys = list(f.keys())
        if max_episodes is not None:
            episode_keys = episode_keys[:max_episodes]

        print(f"Converting {len(episode_keys)} episodes...")

        for ep_idx, ep_key in enumerate(tqdm(episode_keys)):
            ep = f[ep_key]

            # --- Load raw data ---
            rgb_base   = np.array(ep["obs/sensor_data/base_camera/rgb"])   # (T, H, W, 3)
            rgb_wrist  = np.array(ep["obs/sensor_data/hand_camera/rgb"])   # (T, H, W, 3)
            tcp_pose   = np.array(ep["obs/extra/tcp_pose"])                 # (T, 7) [x,y,z,qx,qy,qz,qw]
            qpos       = np.array(ep["obs/agent/qpos"])                     # (T, 9)
            actions    = np.array(ep["actions"])                            # (T-1, 7)

            T = rgb_base.shape[0]  # number of obs steps (T = actions + 1)

            # --- Cartesian position: xyz + euler from tcp_pose ---
            xyz        = tcp_pose[:, :3]                                    # (T, 3)
            quat_xyzw  = tcp_pose[:, 3:]                                    # (T, 4) [qx,qy,qz,qw]
            euler      = np.stack([quat_to_euler(q) for q in quat_xyzw])   # (T, 3)
            cartesian  = np.concatenate([xyz, euler], axis=-1).astype(np.float32)  # (T, 6)

            # --- Gripper position: last joint of qpos ---
            gripper    = qpos[:, -1:].astype(np.float32)                   # (T, 1)

            # --- Actions: split into cartesian_velocity (6D) + gripper_velocity (1D) ---
            # ManiSkill pd_ee_delta_pose: [dx,dy,dz,droll,dpitch,dyaw,gripper]
            # We align obs and actions: obs[t] → action[t] (T-1 actions, pad last)
            act_cart   = actions[:, :6].astype(np.float32)                 # (T-1, 6)
            act_grip   = actions[:, 6:].astype(np.float32)                 # (T-1, 1)
            # Pad last timestep with zeros
            act_cart   = np.concatenate([act_cart, np.zeros((1, 6), dtype=np.float32)], axis=0)  # (T, 6)
            act_grip   = np.concatenate([act_grip, np.zeros((1, 1), dtype=np.float32)], axis=0)  # (T, 1)

            # --- Prompts ---
            prompts    = np.array([task_description] * T)

            # --- Write traj.hdf5 ---
            ep_dir = os.path.join(output_dir, str(ep_idx))
            os.makedirs(ep_dir, exist_ok=True)

            with h5py.File(os.path.join(ep_dir, "traj.hdf5"), "w") as out:
                obs = out.create_group("saved_observation")
                obs.create_dataset("observation/exterior_image_1_left", data=rgb_base,  dtype=np.uint8)
                obs.create_dataset("observation/wrist_image_left",       data=rgb_wrist, dtype=np.uint8)
                obs.create_dataset("observation/cartesian_position",     data=cartesian)
                obs.create_dataset("observation/gripper_position",       data=gripper)
                # Store prompt as fixed-length string
                dt = h5py.string_dtype()
                obs.create_dataset("prompt", data=prompts.astype(bytes), dtype=dt)

                act = out.create_group("action")
                act.create_dataset("cartesian_velocity", data=act_cart)
                act.create_dataset("gripper_velocity",   data=act_grip)

    print(f"Done. {len(episode_keys)} episodes written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path",        required=True,  help="Path to replayed .h5 file")
    parser.add_argument("--output-dir",       required=True,  help="Output directory")
    parser.add_argument("--task-description", required=True,  help="Language instruction")
    parser.add_argument("--max-episodes",     type=int, default=None)
    args = parser.parse_args()

    convert(
        traj_path=args.traj_path,
        output_dir=args.output_dir,
        task_description=args.task_description,
        max_episodes=args.max_episodes,
    )
