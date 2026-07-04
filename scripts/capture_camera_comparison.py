"""
Empirically capture the two cameras discussed with Jesse:
  - the SENSOR camera (what the policy actually receives as observation)
  - the HUMAN RENDER camera (what --save-video / render_rgb_array shows)
and save them side by side as PNGs, with camera IDs and resolutions printed
explicitly — proof, not just a code citation.

Usage:
    python scripts/capture_camera_comparison.py --env-id PickCube-v1
    python scripts/capture_camera_comparison.py --env-id StackCube-v1
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


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.array(x)


def main(env_id, sim_backend, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, obs_mode="rgb", sim_backend=sim_backend, num_envs=1)
    obs, _ = env.reset(seed=0)

    print(f"Sensor cameras available: {list(obs['sensor_data'].keys())}")
    for cam_name, cam_data in obs["sensor_data"].items():
        rgb = to_np(cam_data["rgb"])[0]
        print(f"  '{cam_name}' sensor camera shape: {rgb.shape}")
        path = out_dir / f"{env_id}_sensor_{cam_name}.png"
        iio.imwrite(path, rgb.astype(np.uint8))
        print(f"  -> saved {path}")

    # Human render camera(s) — what --save-video / render_rgb_array shows.
    # Names/resolutions are whatever _default_human_render_camera_configs()
    # defines for this task (e.g. "render_camera", 512x512 for PickCube/StackCube).
    human_cams = list(env.unwrapped._human_render_cameras.keys())
    print(f"Human render cameras available: {human_cams}")
    render_img = env.unwrapped.render_rgb_array()
    render_img = to_np(render_img)[0] if render_img.ndim == 4 else to_np(render_img)
    print(f"  human render camera shape: {render_img.shape}")
    path = out_dir / f"{env_id}_human_render.png"
    iio.imwrite(path, render_img.astype(np.uint8))
    print(f"  -> saved {path}")

    env.close()
    print(f"\nDone. Compare the files in {out_dir}/ directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--out-dir", default="logs/camera_comparison")
    args = parser.parse_args()
    main(args.env_id, args.sim_backend, args.out_dir)
