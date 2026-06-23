"""
Convert ManiSkill trajectory .h5 (with RGB obs) to LeRobot format for π₀.₅ SFT.

Usage:
    python scripts/convert_maniskill_to_lerobot.py \
        --traj-path demos/StackCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
        --repo-name expo_ft/stack_cube \
        --task-description "stack the red cube on top of the green cube" \
        --max-episodes 813
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


IMAGE_SIZE = (128, 128)  # ManiSkill default


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def main(
    traj_path: str,
    repo_name: str,
    task_description: str,
    max_episodes: int = None,
    fps: int = 10,
):
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
                "shape": (*IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (*IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (*IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "cartesian_position": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["cartesian_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
    )

    black = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)

    with h5py.File(traj_path, "r") as f:
        episode_keys = list(f.keys())
        if max_episodes is not None:
            episode_keys = episode_keys[:max_episodes]

        print(f"Converting {len(episode_keys)} episodes to LeRobot format...")

        for ep_key in tqdm(episode_keys):
            ep = f[ep_key]

            rgb_base  = np.array(ep["obs/sensor_data/base_camera/rgb"])   # (T, H, W, 3)
            rgb_wrist = np.array(ep["obs/sensor_data/hand_camera/rgb"])   # (T, H, W, 3)
            tcp_pose  = np.array(ep["obs/extra/tcp_pose"])                 # (T, 7)
            qpos      = np.array(ep["obs/agent/qpos"])                     # (T, 9)
            actions   = np.array(ep["actions"])                            # (T-1, 7)

            T = rgb_base.shape[0]

            # Cartesian position: xyz + euler
            xyz   = tcp_pose[:, :3]
            euler = np.stack([quat_to_euler(q) for q in tcp_pose[:, 3:]])
            cartesian = np.concatenate([xyz, euler], axis=-1).astype(np.float32)

            # Gripper
            gripper = qpos[:, -1:].astype(np.float32)

            # Actions: pad last step with zeros
            actions_padded = np.concatenate(
                [actions, np.zeros((1, 7), dtype=np.float32)], axis=0
            )  # (T, 7)

            for t in range(T):
                dataset.add_frame({
                    "exterior_image_1_left": rgb_base[t],
                    "exterior_image_2_left": black,
                    "wrist_image_left": rgb_wrist[t],
                    "cartesian_position": cartesian[t],
                    "gripper_position": gripper[t],
                    "actions": actions_padded[t],
                    "task": task_description,
                })

            dataset.save_episode()

    print(f"Done. Dataset saved to {output_path}")
    print(f"Total episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path",        required=True)
    parser.add_argument("--repo-name",        required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--max-episodes",     type=int, default=None)
    parser.add_argument("--fps",              type=int, default=10)
    args = parser.parse_args()

    main(
        traj_path=args.traj_path,
        repo_name=args.repo_name,
        task_description=args.task_description,
        max_episodes=args.max_episodes,
        fps=args.fps,
    )
