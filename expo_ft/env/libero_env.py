"""
LIBERO environment wrapper for EXPO-FT.
Drop-in replacement for ManiSkillEnvWrapper using a local LIBERO env.

Supports any LIBERO task suite via cfg.task_suite_name and cfg.task_id.
Modeled after maniskill_env.py — same interface, same config-driven design.

YAML keys used:
    env_id: str           — task suite name (e.g. "libero_spatial", "libero_goal")
    task_id: int          — task index within the suite (0-indexed)
    language_instruction: str  — overrides suite language if set
    max_episode_steps: int
    camera_width: int
    camera_height: int
    state_obs_key: str    — "observation/cartesian_position" (cartesian mode)
    state_obs_dim: int    — dimension of state vector
"""

import logging
import os
import numpy as np

import gymnasium as gym


class LiberoEnvWrapper:
    """Single-env LIBERO wrapper matching the ManiSkillEnvWrapper interface."""

    def __init__(self, env_creation_request: dict, cfg=None):
        self.cfg = cfg
        self.task_suite_name = cfg.env_id          # e.g. "libero_spatial"
        self.task_id         = getattr(cfg, 'task_id', 0)
        self._video_dir      = env_creation_request.get("video_dir", None)
        self._env_usage      = env_creation_request.get("env_usage", "train")

        camera_width  = getattr(cfg, 'camera_width',  128)
        camera_height = getattr(cfg, 'camera_height', 128)

        # Build LIBERO env
        from libero.libero.benchmark import get_benchmark_dict
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = get_benchmark_dict()
        task_suite    = benchmark_dict[self.task_suite_name]()
        task          = task_suite.get_task(self.task_id)
        self._task_language = task.language

        # Language instruction: YAML overrides task language if set
        self.task_description = getattr(cfg, 'language_instruction', None) or self._task_language

        env_args = {
            "bddl_file_name": os.path.join(
                task_suite.get_task_bddl_file_path(self.task_id)
            ),
            "camera_heights": camera_height,
            "camera_widths":  camera_width,
            "camera_names":   ["agentview", "robot0_eye_in_hand"],
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(getattr(cfg, 'seed', 42))

        # Video recording
        if self._video_dir is not None:
            os.makedirs(self._video_dir, exist_ok=True)
        self._frames       = []
        self._episode_count = 0

        self._obs     = None
        self._info    = {}
        self._done    = False
        self._success = False
        self._reward  = 0.0

        logging.info(f"LiberoEnvWrapper: {self.task_suite_name}[{self.task_id}] ({self._env_usage})")

    def reset(self):
        # Save previous episode video if any
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            iio.imwrite(path, self._frames, fps=10, codec='libx264')
            self._frames = []
            self._episode_count += 1

        obs = self._env.reset()
        # Set initial state from task suite
        from libero.libero.benchmark import get_benchmark_dict
        benchmark_dict = get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        init_states = task_suite.get_task_init_states(self.task_id)
        self._env.set_init_state(init_states[0])

        self._obs     = self._parse_obs(obs)
        self._done    = False
        self._success = False
        self._reward  = 0.0
        return self._obs

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        obs, reward, done, info = self._env.step(action)

        if self._video_dir is not None:
            frame = obs.get("agentview_image")
            if frame is not None:
                self._frames.append(np.flipud(frame).astype(np.uint8))

        self._obs     = self._parse_obs(obs)
        self._reward  = float(reward)
        self._done    = bool(done)
        self._success = bool(info.get("success", False))
        self._info    = info
        return action, "policy"

    def get_observation(self):
        return self._obs

    def get_info_for_step(self):
        mask = 1.0 - float(self._done)
        return self._done, self._success, self._reward, mask

    def _parse_obs(self, obs):
        """
        Convert LIBERO obs dict to EXPO-FT format.

        LIBERO keys:
            obs['agentview_image']          (H, W, 3) uint8 — upside down
            obs['robot0_eye_in_hand_image'] (H, W, 3) uint8 — upside down
            obs['robot0_eef_pos']           (3,)
            obs['robot0_eef_quat']          (4,) xyzw
            obs['robot0_gripper_qpos']      (2,)

        EXPO-FT expected keys:
            observation/exterior_image_1_left  (H, W, 3) uint8
            observation/wrist_image_left       (H, W, 3) uint8
            observation/cartesian_position     (6,) [xyz + euler]  — cartesian mode
            observation/gripper_position       (1,)
            prompt                             str
        """
        from scipy.spatial.transform import Rotation

        state_obs_key = getattr(self.cfg, 'state_obs_key', 'observation/cartesian_position')

        # Images — LIBERO uses OpenGL (upside down), flip vertically
        rgb_base  = np.flipud(obs["agentview_image"]).astype(np.uint8)
        rgb_wrist = np.flipud(obs["robot0_eye_in_hand_image"]).astype(np.uint8)

        # State
        eef_pos  = obs["robot0_eef_pos"].astype(np.float32)   # (3,)
        eef_quat = obs["robot0_eef_quat"].astype(np.float32)  # (4,) xyzw

        if state_obs_key == 'observation/cartesian_position':
            euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
            state = np.concatenate([eef_pos, euler]).astype(np.float32)  # (6,)
        elif state_obs_key == 'observation/joint_position':
            state = obs["robot0_joint_pos"].astype(np.float32)  # (7,)
        else:
            euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
            state = np.concatenate([eef_pos, euler]).astype(np.float32)

        # Gripper: mean of two fingers
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
