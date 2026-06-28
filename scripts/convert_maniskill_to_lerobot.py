"""
Convert ManiSkill trajectory .h5 (with RGB obs) to LeRobot format for π₀.₅ SFT.
Config-driven: reads control mode, state obs key, action dims from task YAML.

Usage:
    python scripts/convert_maniskill_to_lerobot.py \
        --traj-path demos/StackCube-v1/rl/trajectory.rgb.pd_joint_delta_pos.physx_cuda.h5 \
        --repo-name expo_ft/stack_cube \
        --task-description "stack the red cube on top of the green cube" \
        --config configs/task/maniskill_stack_cube.yaml
"""

import argparse
import shutil
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import os
os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).resolve().parent.parent / "demos" / "lerobot")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def main(
    traj_path: str,
    repo_name: str,
    task_description: str,
    cfg=None,
    max_episodes: int = None,
    fps: int = 10,
):
    # Config-driven dimensions
    state_obs_key  = getattr(cfg, 'state_obs_key',     'observation/cartesian_position')
    state_obs_dim  = getattr(cfg, 'state_obs_dim',     6)
    action_dim     = getattr(cfg, 'output_action_dim', 7)
    camera_width   = getattr(cfg, 'camera_width',      128)
    camera_height  = getattr(cfg, 'camera_height',     128)
    image_size     = (camera_height, camera_width)
    arm_action_dim = action_dim - 1

    # State feature name (strip "observation/" prefix for LeRobot feature key)
    state_feature_key = state_obs_key.replace("observation/", "")  # e.g. "joint_position"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=fps,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (*image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (*image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (*image_size, 3),
                "names": ["height", "width", "channel"],
            },
            state_feature_key: {
                "dtype": "float32",
                "shape": (state_obs_dim,),
                "names": [state_feature_key],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
    )

    black = np.zeros((*image_size, 3), dtype=np.uint8)

    with h5py.File(traj_path, "r") as f:
        episode_keys = list(f.keys())
        if max_episodes is not None:
            episode_keys = episode_keys[:max_episodes]

        print(f"Converting {len(episode_keys)} episodes to LeRobot format...")
        print(f"  state_obs_key={state_obs_key}, state_obs_dim={state_obs_dim}")
        print(f"  action_dim={action_dim}, image_size={image_size}")

        for ep_key in tqdm(episode_keys):
            ep = f[ep_key]

            rgb_base  = np.array(ep["obs/sensor_data/base_camera/rgb"])   # (T, H, W, 3)
            rgb_wrist = np.array(ep["obs/sensor_data/hand_camera/rgb"])   # (T, H, W, 3)
            tcp_pose  = np.array(ep["obs/extra/tcp_pose"])                 # (T, 7)
            qpos      = np.array(ep["obs/agent/qpos"])                     # (T, 9)
            actions   = np.array(ep["actions"])                            # (T-1, action_dim)

            T = rgb_base.shape[0]

            # State observation
            if state_obs_key == 'observation/joint_position':
                state = qpos[:, :state_obs_dim].astype(np.float32)        # (T, 7)
            else:
                xyz       = tcp_pose[:, :3]
                euler     = np.stack([quat_to_euler(q) for q in tcp_pose[:, 3:]])
                state     = np.concatenate([xyz, euler], axis=-1).astype(np.float32)  # (T, 6)

            # Gripper position: first finger (both fingers always identical on Panda)
            gripper = qpos[:, 7:8].astype(np.float32)                     # (T, 1)

            # Actions: pad last timestep with zeros
            actions_padded = np.concatenate(
                [actions, np.zeros((1, action_dim), dtype=np.float32)], axis=0
            )                                                              # (T, action_dim)

            for t in range(T):
                dataset.add_frame({
                    "exterior_image_1_left": rgb_base[t],
                    "exterior_image_2_left": black,
                    "wrist_image_left":      rgb_wrist[t],
                    state_feature_key:       state[t],
                    "gripper_position":      gripper[t],
                    "actions":               actions_padded[t],
                    "task":                  task_description,
                })

            dataset.save_episode()

    print(f"Done. Dataset saved to {output_path}")
    print(f"Total episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from expo_ft.utils.config_loader import load_task_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path",        required=True)
    parser.add_argument("--repo-name",        required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--config",           default="configs/task/maniskill_stack_cube.yaml")
    parser.add_argument("--max-episodes",     type=int, default=None)
    parser.add_argument("--fps",              type=int, default=10)
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    main(
        traj_path=args.traj_path,
        repo_name=args.repo_name,
        task_description=args.task_description,
        cfg=cfg,
        max_episodes=args.max_episodes,
        fps=args.fps,
    )
