"""
Convert RoboCasa LeRobot dataset to EXPO-FT DROID format.

Usage:
    python scripts/convert_robocasa_to_droid.py \
        --dataset-dir third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot \
        --output-dir demos/robocasa/close_drawer/droid_format \
        --config configs/task/robocasa/close_drawer.yaml
"""

import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def load_video_frames(video_path: str, num_frames: int, target_size: tuple) -> np.ndarray:
    """Load frames from a video file and resize to target_size (H, W)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frames.append(frame)
    cap.release()

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))

    return np.stack(frames[:num_frames]).astype(np.uint8)


def convert(dataset_dir: str, output_dir: str, cfg=None, max_episodes: int = None):
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_h = getattr(cfg, 'camera_height', 224)
    camera_w = getattr(cfg, 'camera_width',  224)
    state_obs_key = getattr(cfg, 'state_obs_key', 'observation/cartesian_position')
    action_space  = getattr(cfg, 'action_space', 'cartesian_velocity')
    gripper_space = getattr(cfg, 'gripper_action_space', 'gripper_velocity')

    # Load task descriptions
    tasks = {}
    with open(dataset_dir / 'meta' / 'tasks.jsonl') as f:
        for line in f:
            d = json.loads(line.strip())
            if 'task_index' in d and 'task' in d:
                tasks[d['task_index']] = d['task']

    # Find all episode parquets
    data_dir = dataset_dir / 'data' / 'chunk-000'
    parquets  = sorted(data_dir.glob('episode_*.parquet'))
    if max_episodes:
        parquets = parquets[:max_episodes]

    print(f"Converting {len(parquets)} episodes...")
    print(f"  state_obs_key={state_obs_key}, image_size=({camera_h},{camera_w})")
    print(f"  action keys: {action_space}, {gripper_space}")

    for ep_idx, parquet_path in enumerate(tqdm(parquets)):
        df = pd.read_parquet(parquet_path)
        T  = len(df)

        # Task description
        task_idx  = int(df['task_index'].iloc[0])
        task_desc = tasks.get(task_idx, cfg.language_instruction if cfg else "close the drawer")

        # Actions: 12D → split into arm (6D) + gripper (1D), ignore base+torso
        actions_12d = np.stack(df['action'].values).astype(np.float32)  # (T, 12)
        act_arm     = actions_12d[:, 5:11]   # eef_pos(3D) + eef_rot(3D)
        act_grip    = actions_12d[:, 11:12]  # gripper(1D)

        # No padding needed — RoboCasa actions already have T frames matching T observations

        # State: extract EEF cartesian from observation.state (16D)
        # Layout: base_pos(3) base_quat(4) eef_pos(3) eef_euler(3) gripper(2) = 15... 
        # Actually we recompute from eef_pos + eef_quat already stored
        # Use columns 7:13 if available, otherwise use state as-is
        states_16d = np.stack(df['observation.state'].values).astype(np.float32)  # (T, 16)
        # eef_pos: cols 6:9, eef_quat: cols 9:13
        # State layout: base_pos(0:3), base_rot(3:7), eef_pos_rel(7:10), eef_rot_rel(10:14), gripper(14:16)
        eef_pos  = states_16d[:, 7:10]
        eef_quat = states_16d[:, 10:14]  # xyzw relative to base
        euler    = np.stack([Rotation.from_quat(q).as_euler("xyz") for q in eef_quat])
        state    = np.concatenate([eef_pos, euler], axis=-1).astype(np.float32)  # (T, 6)

        # Gripper position from state
        gripper  = states_16d[:, 14:15].astype(np.float32)  # (T, 1)

        # Load images from videos
        ep_str    = parquet_path.stem  # "episode_000000"
        chunk_str = parquet_path.parent.name  # "chunk-000"

        base_video  = str(dataset_dir / 'videos' / chunk_str / 'observation.images.robot0_agentview_left' / f'{ep_str}.mp4')
        wrist_video = str(dataset_dir / 'videos' / chunk_str / 'observation.images.robot0_eye_in_hand'   / f'{ep_str}.mp4')

        rgb_base  = load_video_frames(base_video,  T, (camera_h, camera_w))  # (T, H, W, 3)
        if os.path.exists(wrist_video):
            rgb_wrist = load_video_frames(wrist_video, T, (camera_h, camera_w))
        else:
            rgb_wrist = np.zeros_like(rgb_base)
            rgb_wrist[:, 0, 0] = 2  # uint8 sanity check

        prompts = np.array([task_desc] * T)

        # Save
        ep_dir = output_dir / str(ep_idx)
        ep_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(ep_dir / 'traj.hdf5', 'w') as out:
            obs = out.create_group('saved_observation')
            obs.create_dataset('observation/exterior_image_1_left', data=rgb_base,  dtype=np.uint8)
            obs.create_dataset('observation/wrist_image_left',       data=rgb_wrist, dtype=np.uint8)
            obs.create_dataset(state_obs_key,                        data=state)
            obs.create_dataset('observation/gripper_position',       data=gripper)
            dt = h5py.string_dtype()
            obs.create_dataset('prompt', data=prompts.astype(bytes), dtype=dt)

            act = out.create_group('action')
            act.create_dataset(action_space,  data=act_arm)
            act.create_dataset(gripper_space, data=act_grip)

    print(f"Done. {len(parquets)} episodes written to {output_dir}/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from expo_ft.utils.config_loader import load_task_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--config",      default="configs/task/robocasa/close_drawer.yaml")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_task_config(args.config)
    convert(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        max_episodes=args.max_episodes,
    )
