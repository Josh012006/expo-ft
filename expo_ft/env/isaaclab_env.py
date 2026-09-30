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

        logging.info(f"IsaacLabEnvWrapper: connected, task={self.task_description!r}")

    def reset(self, **reset_kwargs):
        """reset_kwargs: currently only `seed` is forwarded to the server —
        see env_client.EnvClient.reset for the wire-protocol side of this."""
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
        return self._client.get_observation()

    def get_info_for_step(self):
        mask = 1.0 - float(self._done)
        return self._done, self._success, self._reward, mask

    def get_raw_info(self):
        return dict(self._info)

    def close(self):
        pass  # the server owns the environment lifecycle, not this wrapper
