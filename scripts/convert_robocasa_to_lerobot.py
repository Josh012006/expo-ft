"""
Convert RoboCasa LeRobot dataset to EXPO-FT LeRobot format.
Renames keys to match our pipeline (action→actions, etc.)

Usage:
    python scripts/convert_robocasa_to_lerobot.py \
        --dataset-dir third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot \
        --repo-name expo_ft/robocasa_close_drawer \
        --config configs/task/robocasa/close_drawer.yaml
"""

import argparse, json, os, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).resolve().parent.parent / "demos" / "lerobot")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def load_video_frames(video_path, num_frames, target_size):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
    return np.stack(frames[:num_frames]).astype(np.uint8)


def main(dataset_dir, repo_name, cfg, max_episodes=None, fps=20):
    dataset_dir = Path(dataset_dir)
    camera_h = getattr(cfg, 'camera_height', 224)
    camera_w = getattr(cfg, 'camera_width', 224)
    state_obs_key = getattr(cfg, 'state_obs_key', 'observation/cartesian_position')
    state_obs_dim = getattr(cfg, 'state_obs_dim', 6)

    # Load task descriptions
    tasks = {}
    with open(dataset_dir / 'meta' / 'tasks.jsonl') as f:
        for line in f:
            d = json.loads(line.strip())
            if 'task_index' in d and 'task' in d:
                tasks[d['task_index']] = d['task']

    state_feature_key = state_obs_key.replace("observation/", "")

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=fps,
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": (camera_h, camera_w, 3), "names": ["height", "width", "channel"]},
            "wrist_image_left":      {"dtype": "image", "shape": (camera_h, camera_w, 3), "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "image", "shape": (camera_h, camera_w, 3), "names": ["height", "width", "channel"]},
            state_feature_key:       {"dtype": "float32", "shape": (state_obs_dim,), "names": [state_feature_key]},
            "gripper_position":      {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions":               {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
    )

    data_dir = dataset_dir / 'data' / 'chunk-000'
    parquets = sorted(data_dir.glob('episode_*.parquet'))
    if max_episodes:
        parquets = parquets[:max_episodes]

    print(f"Converting {len(parquets)} episodes to LeRobot format...")
    black = np.zeros((camera_h, camera_w, 3), dtype=np.uint8)

    for parquet_path in tqdm(parquets):
        df = pd.read_parquet(parquet_path)
        T = len(df)
        ep_str = parquet_path.stem
        chunk_str = parquet_path.parent.name

        task_idx = int(df['task_index'].iloc[0])
        task_desc = tasks.get(task_idx, cfg.language_instruction)

        # Actions: eef(5:11) + gripper(11:12)
        actions_12d = np.stack(df['action'].values).astype(np.float32)
        act_arm  = actions_12d[:, 5:11]
        act_grip = actions_12d[:, 11:12]
        actions  = np.concatenate([act_arm, act_grip], axis=-1)  # (T, 7)

        # State: eef_pos_rel(7:10) + eef_rot_rel(10:14) → euler
        states_16d = np.stack(df['observation.state'].values).astype(np.float32)
        eef_pos  = states_16d[:, 7:10]
        eef_quat = states_16d[:, 10:14]
        euler = np.stack([Rotation.from_quat(q).as_euler("xyz") for q in eef_quat])
        state = np.concatenate([eef_pos, euler], axis=-1).astype(np.float32)
        gripper = states_16d[:, 14:15].astype(np.float32)

        # Images
        base_video  = str(dataset_dir / 'videos' / chunk_str / 'observation.images.robot0_agentview_left' / f'{ep_str}.mp4')
        wrist_video = str(dataset_dir / 'videos' / chunk_str / 'observation.images.robot0_eye_in_hand'   / f'{ep_str}.mp4')
        rgb_base  = load_video_frames(base_video, T, (camera_h, camera_w))
        rgb_wrist = load_video_frames(wrist_video, T, (camera_h, camera_w)) if os.path.exists(wrist_video) else np.zeros((T, camera_h, camera_w, 3), dtype=np.uint8)

        for t in range(T):
            dataset.add_frame({
                "exterior_image_1_left": rgb_base[t],
                "wrist_image_left":      rgb_wrist[t],
                "exterior_image_2_left": black,
                state_feature_key:       state[t],
                "gripper_position":      gripper[t],
                "actions":               actions[t],
                "task":                  task_desc,
            })
        dataset.save_episode()

    print(f"Done. Dataset saved to {output_path}")
    print(f"Total episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from expo_ft.utils.config_loader import load_task_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-name",   required=True)
    parser.add_argument("--config",      default="configs/task/robocasa/close_drawer.yaml")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps",          type=int, default=20)
    args = parser.parse_args()
    cfg = load_task_config(args.config)
    main(args.dataset_dir, args.repo_name, cfg, args.max_episodes, args.fps)
