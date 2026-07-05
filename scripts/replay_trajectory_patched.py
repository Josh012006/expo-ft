"""
Thin wrapper around mani_skill.trajectory.replay_trajectory that:
  1. Applies the PickCube visible-goal patch (expo_ft/env/patches.py).
  2. Monkeypatches gym.make so that when replay_trajectory internally creates
     its env (gym.make(env_id, **env_kwargs) inside replay_trajectory.py),
     our own sensor_configs (camera pose/fov/resolution) and robot_uids
     overrides get merged in too — matching exactly what
     expo_ft/env/maniskill_env.py applies for eval/RL. Without this, demo RGB
     conversion silently used ManiSkill's raw defaults (128x128, original
     camera pose, plain "panda" robot with no wrist cam), causing a
     resolution/shape mismatch against the LeRobot dataset schema (which
     expects whatever the task YAML's camera_width/height says).

Why this is a separate process-level patch: replay_trajectory runs as its
own subprocess, so any monkeypatch applied only inside expo_ft's own process
(e.g. maniskill_env.py) never reaches it.

Usage: same CLI args as the original tool, PLUS --expo-config pointing to the
task YAML (consumed here, not passed through to replay_trajectory's parser):
    python scripts/replay_trajectory_patched.py \\
        --expo-config configs/task/maniskill/push_cube.yaml \\
        --traj-path demos/PushCube-v1/motionplanning/trajectory.h5 \\
        --save-traj -o rgb -c pd_joint_delta_pos -b physx_cpu
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.patches import patch_pickcube_visible_goal
patch_pickcube_visible_goal()

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--expo-config", required=True)
_known, _remaining = _pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

from expo_ft.utils.config_loader import load_task_config
from mani_skill.utils.sapien_utils import look_at

_cfg = load_task_config(_known.expo_config)
_eye_pos = getattr(_cfg, 'camera_eye_pos', [0.3, 0, 0.6])
_target_pos = getattr(_cfg, 'camera_target_pos', [-0.1, 0, 0.1])
_camera_pose = look_at(eye=_eye_pos, target=_target_pos).raw_pose[0].cpu().tolist()
_camera_fov = getattr(_cfg, 'camera_fov', 1.0)
_width = getattr(_cfg, 'camera_width', 128)
_height = getattr(_cfg, 'camera_height', 128)
_robot_uids = getattr(_cfg, 'robot_uids', 'panda_wristcam')

import gymnasium as gym
_original_gym_make = gym.make


def _patched_gym_make(id, **kwargs):
    if id == _cfg.env_id:
        kwargs.setdefault("sensor_configs", {})
        kwargs["sensor_configs"] = {
            "width": _width,
            "height": _height,
            "base_camera": {"pose": _camera_pose, "fov": _camera_fov},
            **kwargs["sensor_configs"],
        }
        kwargs.setdefault("robot_uids", _robot_uids)
    return _original_gym_make(id, **kwargs)


gym.make = _patched_gym_make

from mani_skill.trajectory.replay_trajectory import main, parse_args

if __name__ == "__main__":
    main(parse_args())
