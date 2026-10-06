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
import json
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

        # Peg-to-socket diagnostics (FORGE server only): per-episode summary of how close the
        # held peg got to its target. A binary success rate hides everything short of an
        # insertion, which is the whole picture for a zero-shot run.
        self._peg_offsets = []
        self._metrics_episode = 0
        self._metrics_header_printed = False

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

    def _record_metrics(self, obs):
        off = obs.get("extra/peg_socket_offset")
        if off is not None:
            self._peg_offsets.append(np.asarray(off, dtype=np.float64).reshape(-1)[:3])

    def _flush_metrics(self):
        """Print (and append to <video_dir>/peg_socket_metrics.jsonl) the summary of the episode
        just finished. Offset = held peg base minus target, env frame, metres."""
        if not self._peg_offsets:
            return
        o = np.stack(self._peg_offsets)
        xy = np.linalg.norm(o[:, :2], axis=1)
        k = int(np.argmin(xy))
        mm = lambda v: 1000.0 * float(v)
        if not self._metrics_header_printed:
            print("[peg-socket] offset = held peg base - target pose in the socket (env frame). FORGE "
                  "success needs xy < 2.5 mm AND z below its height threshold.", flush=True)
            self._metrics_header_printed = True
        print(
            f"[peg-socket] episode {self._metrics_episode}: start xy={mm(xy[0]):.1f} mm z={mm(o[0, 2]):+.1f} mm"
            f" | closest xy={mm(xy[k]):.1f} mm (z={mm(o[k, 2]):+.1f} mm) at step {k}"
            f" | final x={mm(o[-1, 0]):+.1f} y={mm(o[-1, 1]):+.1f} mm (xy={mm(xy[-1]):.1f}) z={mm(o[-1, 2]):+.1f} mm",
            flush=True,
        )
        if self._video_dir is not None:
            record = {
                "episode": self._metrics_episode, "steps": int(len(o)),
                "start_xy_mm": mm(xy[0]), "closest_xy_mm": mm(xy[k]), "closest_step": k,
                "closest_z_mm": mm(o[k, 2]), "final_offset_mm": [mm(v) for v in o[-1]],
                "final_xy_mm": mm(xy[-1]),
            }
            with open(os.path.join(self._video_dir, "peg_socket_metrics.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        self._peg_offsets = []
        self._metrics_episode += 1

    def reset(self, **reset_kwargs):
        """reset_kwargs: currently only `seed` is forwarded to the server —
        see env_client.EnvClient.reset for the wire-protocol side of this."""
        self._flush_video()
        self._flush_metrics()
        obs = self._client.reset(seed=reset_kwargs.get("seed"))
        self._record_metrics(obs)
        self._done = False
        self._success = False
        self._reward = 0.0
        return obs

    def step(self, action):
        """Returns (real_executed_action, action_type), matching
        ManiSkillEnvWrapper.step. `action` is 8D: 7 ABSOLUTE joint targets
        (rad) + the gripper command (0 = open, 1 = closed). The model's raw
        chunk is a set of OFFSETS from the joint state at plan time; the
        caller converts it once per chunk (scripts/eval_policy.py,
        `chunk_action_reference`) — nothing is added to the live state here
        or on the server."""
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
        self._record_metrics(obs)
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
        self._flush_metrics()
