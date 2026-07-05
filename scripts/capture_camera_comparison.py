"""
Empirically capture the two cameras discussed with Jesse:
  - the SENSOR camera (what the policy actually receives as observation)
  - the HUMAN RENDER camera (what --save-video / render_rgb_array shows)
and save them side by side as PNGs, with camera IDs and resolutions printed
explicitly — proof, not just a code citation.

Loads the SAME task YAML and applies the SAME sensor_configs override
(width/height + base_camera pose) as expo_ft/env/maniskill_env.py, so this
actually reflects what the real training/eval pipeline sees — not
ManiSkill's raw defaults.

Usage:
    python scripts/capture_camera_comparison.py --config configs/task/maniskill/pick_cube.yaml
    python scripts/capture_camera_comparison.py --config configs/task/maniskill/stack_cube.yaml
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: registers env ids
import imageio.v3 as iio
from mani_skill.utils.sapien_utils import look_at

from expo_ft.utils.config_loader import load_task_config


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.array(x)


def main(config_path, out_dir):
    cfg = load_task_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eye_pos = getattr(cfg, 'camera_eye_pos', [0.3, 0, 0.6])
    target_pos = getattr(cfg, 'camera_target_pos', [-0.1, 0, 0.1])
    camera_pose = look_at(eye=eye_pos, target=target_pos).raw_pose[0].cpu().tolist()
    width = getattr(cfg, 'camera_width', 128)
    height = getattr(cfg, 'camera_height', 128)

    print(f"Using camera_eye_pos={eye_pos}, camera_target_pos={target_pos}, "
          f"width={width}, height={height} (from {config_path})")

    env = gym.make(
        cfg.env_id,
        obs_mode="rgb",
        sim_backend=getattr(cfg, 'sim_backend', 'physx_cpu'),
        num_envs=1,
        robot_uids=getattr(cfg, 'robot_uids', 'panda_wristcam'),
        sensor_configs=dict(
            width=width,
            height=height,
            base_camera=dict(pose=camera_pose),
        ),
    )
    obs, _ = env.reset(seed=0)

    print(f"Sensor cameras available: {list(obs['sensor_data'].keys())}")
    for cam_name, cam_data in obs["sensor_data"].items():
        rgb = to_np(cam_data["rgb"])[0]
        print(f"  '{cam_name}' sensor camera shape: {rgb.shape}")
        path = out_dir / f"{cfg.env_id}_sensor_{cam_name}.png"
        iio.imwrite(path, rgb.astype(np.uint8))
        print(f"  -> saved {path}")

    human_cams = list(env.unwrapped._human_render_cameras.keys())
    print(f"Human render cameras available: {human_cams}")
    render_img = env.unwrapped.render_rgb_array()
    render_img = to_np(render_img)[0] if render_img.ndim == 4 else to_np(render_img)
    print(f"  human render camera shape: {render_img.shape}")
    path = out_dir / f"{cfg.env_id}_human_render.png"
    iio.imwrite(path, render_img.astype(np.uint8))
    print(f"  -> saved {path}")

    env.close()
    print(f"\nDone. Compare the files in {out_dir}/ directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to task YAML config.")
    parser.add_argument("--out-dir", default="logs/camera_comparison")
    args = parser.parse_args()
    main(args.config, args.out_dir)
