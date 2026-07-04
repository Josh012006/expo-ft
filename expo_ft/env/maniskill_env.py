"""
ManiSkill environment wrapper that reproduces the EnvClientWrapper interface
for use with the EXPO-FT training loop.
"""

import logging
import numpy as np
import os

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.sapien_utils import look_at


class ManiSkillEnvWrapper:
    """Drop-in replacement for EnvClientWrapper using a local ManiSkill env."""

    def __init__(self, env_creation_request: dict, cfg=None):
        """
        Args:
            env_creation_request: dict with keys:
                - example_action: np.ndarray
                - env_usage: str ("train" or "eval")
                - video_dir: str (path to save videos, or None)
            cfg: task config (loaded from YAML)
        """
        self.cfg = cfg
        self.env_id = cfg.env_id
        self.task_description = cfg.language_instruction
        self._video_dir = env_creation_request.get("video_dir", None)
        self._env_usage = env_creation_request.get("env_usage", "train")

        eye_pos = getattr(cfg, 'camera_eye_pos', [0.3, 0, 0.6])
        target_pos = getattr(cfg, 'camera_target_pos', [-0.1, 0, 0.1])
        camera_pose = look_at(eye=eye_pos, target=target_pos).raw_pose[0].cpu().tolist()

        self._env = gym.make(
            cfg.env_id,
            obs_mode="rgb",
            control_mode=cfg.control_mode,
            num_envs=1,
            max_episode_steps=cfg.max_episode_steps,
            sensor_configs=dict(
                width=getattr(cfg, 'camera_width', 128),
                height=getattr(cfg, 'camera_height', 128),
                base_camera=dict(pose=camera_pose),
            ),
            sim_backend=getattr(cfg, 'sim_backend', 'physx_cuda'),
        )

        # Manual video recording (gymnasium RecordVideo incompatible with ManiSkill tensors)
        if self._video_dir is not None:
            os.makedirs(self._video_dir, exist_ok=True)
        self._frames = []
        self._episode_count = 0

        self._obs = None
        self._info = {}
        self._done = False
        self._success = False
        self._reward = 0.0

        logging.info(f"ManiSkillEnvWrapper: created {cfg.env_id} ({self._env_usage})")

    def reset(self, **reset_kwargs):
        """Reset the environment and return observation.

        reset_kwargs are forwarded to the underlying ManiSkill env's reset() —
        e.g. seed=..., options=... — to allow reproducing a specific recorded
        episode exactly. Defaults to no kwargs (existing random-reset behavior
        for eval/RL is unchanged).
        """
        # Save previous episode video if any
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            iio.imwrite(path, self._frames, fps=10, codec='libx264')
            self._frames = []
            self._episode_count += 1

        obs, info = self._env.reset(**reset_kwargs)
        self._obs = self._parse_obs(obs)
        self._info = info
        self._done = False
        self._success = False
        self._reward = 0.0
        return self._obs

    def step(self, action):
        """
        Step the environment.

        Returns:
            (real_executed_action, action_type)
            action_type is always "policy" (no human intervention in sim)
        """
        action = np.array(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._video_dir is not None:
            # Tile every available sensor camera side by side (e.g. base_camera +
            # hand_camera when present) instead of only showing base_camera.
            cam_frames = []
            for cam_name in sorted(obs['sensor_data'].keys()):
                frame = obs['sensor_data'][cam_name]['rgb']
                if hasattr(frame, 'cpu'):
                    frame = frame.cpu().numpy()
                cam_frames.append(np.array(frame[0]).astype(np.uint8))
            tiled = tile_images(cam_frames) if len(cam_frames) > 1 else cam_frames[0]
            self._frames.append(tiled)
        self._obs = self._parse_obs(obs)
        self._reward = float(reward.item() if hasattr(reward, 'item') else reward)
        self._done = bool((terminated | truncated).item()
                          if hasattr(terminated, 'item') else (terminated or truncated))
        self._success = bool(info.get("success", False))
        if hasattr(self._success, 'item'):
            self._success = self._success.item()
        self._info = info
        return action, "policy"

    def get_observation(self):
        """Return the current observation."""
        return self._obs

    def get_info_for_step(self):
        """
        Return (done, success, reward, mask).
        mask = 1 - done (continuation mask for RL).
        """
        mask = 1.0 - float(self._done)
        return self._done, self._success, self._reward, mask

    def get_raw_info(self):
        """Return the full raw info dict from the last step() call — e.g.
        is_obj_placed, is_robot_static, is_grasped for tasks that expose them.
        Purely additive/diagnostic: does not affect get_info_for_step() or the
        success value used anywhere else in the pipeline."""
        return dict(self._info) if self._info is not None else {}

    def _parse_obs(self, obs):
        """
        Convert ManiSkill obs dict to the format expected by the EXPO-FT replay buffer.

        ManiSkill keys:
            obs['sensor_data']['base_camera']['rgb']  (1, H, W, 3)
            obs['sensor_data']['hand_camera']['rgb']  (1, H, W, 3)
            obs['agent']['qpos']                      (1, 9)
            obs['extra']['tcp_pose']                  (1, 7)

        EXPO-FT expected keys (matching DroidInputs):
            observation/exterior_image_1_left  (H, W, 3) uint8
            observation/wrist_image_left       (H, W, 3) uint8
            observation/cartesian_position     (6,) float32
            observation/gripper_position       (1,) float32
            prompt                             str
        """
        from scipy.spatial.transform import Rotation

        def to_np(t):
            return t.cpu().numpy() if hasattr(t, 'cpu') else np.array(t)

        rgb_base  = to_np(obs['sensor_data']['base_camera']['rgb'])[0]   # (H, W, 3)
        if 'hand_camera' in obs['sensor_data']:
            rgb_wrist = to_np(obs['sensor_data']['hand_camera']['rgb'])[0]   # (H, W, 3)
        else:
            # No wrist camera available for this task (e.g. PushCube) — use a near-black image.
            # (one pixel set to 1 to satisfy the uint8 sanity check downstream: max > 1)
            rgb_wrist = np.zeros_like(rgb_base)
            rgb_wrist[0, 0] = 2
        tcp_pose  = to_np(obs['extra']['tcp_pose'])[0]                   # (7,)
        qpos      = to_np(obs['agent']['qpos'])[0]                       # (9,)

        state_key = getattr(self.cfg, 'state_obs_key', 'observation/cartesian_position')
        if state_key == 'observation/joint_position':
            # 7 joint positions from qpos
            state = qpos[:7].astype(np.float32)
        elif state_key == 'observation/cartesian_position':
            # Cartesian position: xyz + euler from quaternion
            xyz        = tcp_pose[:3]
            quat_xyzw  = tcp_pose[3:]
            euler      = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
            state      = np.concatenate([xyz, euler]).astype(np.float32)

        # Gripper: last joint
        gripper = qpos[-1:].astype(np.float32)

        return {
            "observation/exterior_image_1_left": rgb_base.astype(np.uint8),
            "observation/wrist_image_left":      rgb_wrist.astype(np.uint8),
            state_key:    state,
            "observation/gripper_position":      gripper,
            "prompt":                            self.task_description,
        }

    def close(self):
        # Save last episode video
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            iio.imwrite(path, self._frames, fps=10, codec='libx264')
        self._env.close()
