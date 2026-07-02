"""
RoboCasa environment wrapper for EXPO-FT.
Drop-in replacement for ManiSkillEnvWrapper using a local RoboCasa env.

Supports any RoboCasa task via cfg.env_id.
Modeled after maniskill_env.py — same interface, same config-driven design.

YAML keys used:
    env_id: str                — RoboCasa task name (e.g. "PickPlaceCounterToCabinet")
    language_instruction: str  — task language instruction
    max_episode_steps: int
    camera_width: int
    camera_height: int
    robot_name: str            — robot to use (default: "PandaOmron")
    state_obs_key: str         — observation key for state
    state_obs_dim: int         — state dimension
"""

import logging
import os
import numpy as np
from scipy.spatial.transform import Rotation


class RoboCasaEnvWrapper:
    """Single-env RoboCasa wrapper matching the ManiSkillEnvWrapper interface."""

    def __init__(self, env_creation_request: dict, cfg=None):
        self.cfg = cfg
        self.task_name    = cfg.env_id
        self._video_dir   = env_creation_request.get("video_dir", None)
        self._env_usage   = env_creation_request.get("env_usage", "train")

        camera_width  = getattr(cfg, 'camera_width',  224)
        camera_height = getattr(cfg, 'camera_height', 224)
        robot_name    = getattr(cfg, 'robot_name', 'PandaOmron')
        seed          = getattr(cfg, 'seed', 42)

        self.task_description = cfg.language_instruction

        # Build RoboCasa env via robosuite
        import robocasa  # registers environments
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller=None,
            robot=robot_name,
        )

        self._env = robosuite.make(
            env_name=self.task_name,
            robots=robot_name,
            controller_configs=controller_config,
            camera_names=[
                "robot0_agentview_left",
                "robot0_eye_in_hand",
                "robot0_agentview_right",
            ],
            camera_widths=camera_width,
            camera_heights=camera_height,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=seed,
            translucent_robot=False,
        )

        # Video recording
        if self._video_dir is not None:
            os.makedirs(self._video_dir, exist_ok=True)
        self._frames        = []
        self._episode_count = 0

        self._obs     = None
        self._info    = {}
        self._done    = False
        self._success = False
        self._reward  = 0.0

        import json
        self._action_q01 = None
        self._action_q99 = None
        self._action_range = None
        if hasattr(cfg, 'action_stats_path') and cfg.action_stats_path:
            with open(cfg.action_stats_path) as f:
                stats = json.load(f)
            self._action_q01 = np.array(stats['q01'], dtype=np.float32)
            self._action_q99 = np.array(stats['q99'], dtype=np.float32)
            self._action_range = self._action_q99 - self._action_q01
            print(f"Loaded action stats from {cfg.action_stats_path}")

        logging.info(f"RoboCasaEnvWrapper: {self.task_name} ({self._env_usage})")

    def reset(self):
        # Save previous episode video if any
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            iio.imwrite(path, self._frames, fps=10, codec='libx264')
            self._frames = []
            self._episode_count += 1

        obs = self._env.reset()
        # Read dynamic language instruction from env
        if hasattr(self._env, '_ep_meta') and self._env._ep_meta.get('lang'):
            self.task_description = self._env._ep_meta['lang']
        self._obs     = self._parse_obs(obs)
        self._done    = False
        self._success = False
        self._reward  = 0.0
        return self._obs

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        # Exact affine mapping from pi0.5 output distribution to [-1, 1]
        if self._action_q01 is not None:
            action = (action - self._action_q01) / self._action_range * 2 - 1
            action = np.clip(action, -1, 1)
        # π₀.₅ produces 7D actions [eef(6) + gripper(1)].
        # PandaOmron expects 12D: [right(6), right_gripper(1), base(3), torso(1)].
        # Fix base and torso to 0 for tasks where the mobile base stays still.
        action_12d = np.zeros(12, dtype=np.float32)
        action_12d[:min(7, len(action))] = action[:7]
        obs, reward, done, info = self._env.step(action_12d)

        if self._video_dir is not None:
            frame = obs.get("robot0_agentview_left_image")
            if frame is not None:
                self._frames.append(np.flipud(frame).astype(np.uint8))

        self._obs     = self._parse_obs(obs)
        self._reward  = float(reward)
        self._success = bool(info.get("success", False))
        self._done    = bool(done) or self._success
        self._info    = info
        return action, "policy"

    def get_observation(self):
        return self._obs

    def get_info_for_step(self):
        mask = 1.0 - float(self._done)
        return self._done, self._success, self._reward, mask

    def _parse_obs(self, obs):
        """
        Convert RoboCasa obs dict to EXPO-FT format.

        RoboCasa keys:
            obs['robot0_agentview_left_image']  (H, W, 3) uint8 — upside down
            obs['robot0_eye_in_hand_image']     (H, W, 3) uint8 — upside down
            obs['robot0_eef_pos']               (3,)
            obs['robot0_eef_quat']              (4,) xyzw
            obs['robot0_gripper_qpos']          (2,)
            obs['robot0_base_pos']              (3,)
            obs['robot0_base_quat']             (4,)

        EXPO-FT expected keys:
            observation/exterior_image_1_left  (H, W, 3) uint8
            observation/wrist_image_left       (H, W, 3) uint8
            observation/cartesian_position     (6,) [xyz + euler]
            observation/gripper_position       (1,)
            prompt                             str
        """
        state_obs_key = getattr(self.cfg, 'state_obs_key', 'observation/cartesian_position')
        state_obs_dim = getattr(self.cfg, 'state_obs_dim', 6)

        # Images — RoboCasa uses OpenGL (upside down), flip vertically
        rgb_base  = np.flipud(obs["robot0_agentview_left_image"]).astype(np.uint8)
        rgb_wrist = np.flipud(obs["robot0_eye_in_hand_image"]).astype(np.uint8)

        # State — use base-relative EEF to match LeRobot dataset convention
        eef_pos  = obs["robot0_base_to_eef_pos"].astype(np.float32)   # (3,) relative to base
        eef_quat = obs["robot0_base_to_eef_quat"].astype(np.float32)  # (4,) xyzw relative to base

        if state_obs_key == 'observation/cartesian_position':
            euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
            state = np.concatenate([eef_pos, euler]).astype(np.float32)  # (6,)
        elif state_obs_key == 'observation/joint_position':
            state = obs["robot0_joint_pos"][:state_obs_dim].astype(np.float32)
        else:
            euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
            state = np.concatenate([eef_pos, euler]).astype(np.float32)

        # Gripper: first finger
        gripper = obs["robot0_gripper_qpos"][:1].astype(np.float32)  # (1,)

        return {
            "observation/exterior_image_1_left": rgb_base,
            "observation/wrist_image_left":      rgb_wrist,
            state_obs_key:                       state,
            "observation/gripper_position":      gripper,
            "prompt":                            self.task_description,
        }

    def close(self):
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            iio.imwrite(path, self._frames, fps=10, codec='libx264')
        self._env.close()
