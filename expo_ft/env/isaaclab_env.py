"""Isaac Lab / FORGE client — same interface as ManiSkillEnvWrapper, for use
with the EXPO-FT training/eval loop (train_pi_robo.py, eval_policy.py, ...).

Unlike ManiSkillEnvWrapper, there's no local simulation here: this class
talks to expo_ft/env/isaaclab/forge_server.py over the exact wire protocol
env_client.EnvClient already speaks for the real DROID robot. That's a
deliberate reuse, not a coincidence — the server owns the physical world
(simulated today, a real Franka later), this wrapper owns nothing but the
connection, so pointing it at a robot's server instead of forge_server.py
should need zero changes here.

The server already emits observations keyed exactly like DroidInputs expects
(observation/exterior_image_1_left, observation/wrist_image_left,
observation/joint_position, observation/gripper_position, prompt) plus an
extra/* namespace for tactile-phase diagnostics not yet consumed anywhere —
so, also unlike ManiSkillEnvWrapper, there is no _parse_obs step here.
"""
import logging
import os

import numpy as np

from expo_ft.env.env_client import EnvClientWrapper


class IsaacLabEnvWrapper:
    """Drop-in replacement for EnvClientWrapper/ManiSkillEnvWrapper, backed
    by an Isaac Lab / FORGE environment server."""

    def __init__(self, env_creation_request: dict, cfg=None):
        self.cfg = cfg
        self._client = EnvClientWrapper(
            env_creation_request,
            host=getattr(cfg, "server_host", "localhost"),
            port=getattr(cfg, "server_port", 8102),
        )
        self.env_id = self._client.env_id
        self.task_description = self._client.task_description
        self._info = {}
        self._done = False
        self._success = False
        self._reward = 0.0

        # Video recording — mirrors ManiSkillEnvWrapper's own pattern (manual
        # frame accumulation + imageio write at episode boundaries), added
        # here because eval_policy.py passes video_dir through
        # env_creation_request expecting every env wrapper to handle it the
        # same way; this one previously just silently ignored it.
        self._video_dir = env_creation_request.get("video_dir", None)
        if self._video_dir is not None:
            os.makedirs(self._video_dir, exist_ok=True)
        self._frames = []
        self._episode_count = 0

        logging.info(f"IsaacLabEnvWrapper: connected, task={self.task_description!r}")

    def _flush_video(self):
        """Write the accumulated frames of the episode just finished. Called
        from reset() (previous episode) and close() (the last one, which
        never gets a following reset() to trigger it)."""
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            iio.imwrite(path, self._frames, fps=10, codec="libx264")
            self._frames = []
            self._episode_count += 1

    def reset(self, **reset_kwargs):
        """reset_kwargs: currently only `seed` is forwarded to the server —
        see env_client.EnvClient.reset for the wire-protocol side of this."""
        self._flush_video()
        obs = self._client.reset(seed=reset_kwargs.get("seed"))
        self._done = False
        self._success = False
        self._reward = 0.0
        return obs

    def step(self, action):
        """Returns (real_executed_action, action_type), matching
        ManiSkillEnvWrapper.step. `action` is the raw 8D output of
        pi05_droid_jointpos (7 joint-position deltas + gripper) — no
        client-side math: the server adds the delta to its own live joint
        state in _apply_action, since it owns the authoritative physics."""
        action = np.array(action, dtype=np.float32)
        real_action, action_type = self._client.step(action)
        done, success, reward, _mask = self._client.get_info_for_step()
        self._done, self._success, self._reward = done, success, reward
        return real_action, action_type

    def get_observation(self):
        """Frame capture is hooked here, not step(): eval_policy.py's loop
        calls step() -> get_info_for_step() -> get_observation() exactly
        once each per env step, so this is called exactly once per frame
        with no extra round-trip and no risk of duplicate frames for that
        calling pattern. (A script that calls get_observation() more than
        once per step with video_dir set would get extra frames — none of
        the current callers do.)"""
        obs = self._client.get_observation()
        if self._video_dir is not None:
            # sim server emits "observation/..." keys, the real-robot server plain keys
            ext = obs.get("observation/exterior_image_1_left", obs.get("exterior_image_1_left"))
            wrist = obs.get("observation/wrist_image_left", obs.get("wrist_image_left"))
            if ext is not None and wrist is not None:
                tiled = np.concatenate(
                    [np.asarray(ext, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)], axis=1
                )  # side by side, same height
                self._frames.append(tiled)
        return obs

    def get_info_for_step(self):
        mask = 1.0 - float(self._done)
        return self._done, self._success, self._reward, mask

    def get_raw_info(self):
        return dict(self._info)

    def close(self):
        self._flush_video()
