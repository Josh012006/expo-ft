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
from expo_ft.env.video_panels import TACTILE_VMAX, compose_frame, tactile_stats, text_banner


class IsaacLabEnvWrapper:
    """Drop-in replacement for EnvClientWrapper/ManiSkillEnvWrapper, backed
    by an Isaac Lab / FORGE environment server."""

    BANNER_H = 48      # height (px) of the metrics banner under the eval video frames. A frame is 448 x 448 (two cameras on
                       # top, the taxel panel below, see video_panels.py) + 48 = 496: a multiple of 16, otherwise the video
                       # encoder rescales the frames
    LAST_N = 20        # "last 20 steps" window of the contact-force / command-gap statistics of an episode

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

        # Success rule (see step()). FORGE never ends an episode on success, it only times out, and the
        # step that times out has ALREADY auto-reset the simulation: the server's success flag read at
        # that step describes the next episode's start state, so it is always False. Reading the flag
        # at the end of the episode, as eval_policy.py does, therefore gave 0 % whatever the policy did.
        # Here an episode counts as a success once the server says "inserted" for `success_hold_steps`
        # steps in a row (same rule and default as the demo collector, HOLD_AFTER_SUCCESS = 5), and
        # it ends at that moment. 0 = legacy behaviour (flag of the last step; always False).
        self._hold_steps = int(getattr(cfg, "success_hold_steps", 5))
        self._server_done = False     # the server really ended the episode (timeout): sim already reset
        self._flag = False            # the server's own "inserted" flag at the last step (shown in the video banner)
        self._streak = 0              # consecutive steps with the server's success flag on
        self._max_streak = 0          # longest streak of the episode

        # Video recording — mirrors ManiSkillEnvWrapper's own pattern (manual
        # frame accumulation + imageio write at episode boundaries), added
        # here because eval_policy.py passes video_dir through
        # env_creation_request expecting every env wrapper to handle it the
        # same way; this one previously just silently ignored it.
        self._video_dir = env_creation_request.get("video_dir", None)
        if self._video_dir is not None:
            os.makedirs(self._video_dir, exist_ok=True)
        self._frames = []
        self._frame_infos = []        # one dict per frame: what the banner under the video frame shows
        self._episode_count = 0

        # Peg-to-socket diagnostics (FORGE server only): per-episode summary of how close the
        # held peg got to its target. A binary success rate hides everything short of an
        # insertion, which is the whole picture for a zero-shot run.
        self._peg_offsets = []
        self._closest = (float("inf"), 0)   # (xy in m, step) of the closest approach so far this episode
        self._metrics_episode = 0
        self._metrics_header_printed = False

        # Contact force and command gap (to tell a physical jam from a policy that stops commanding motion):
        #   force   norm of the first 3 components of extra/ft_force at every observation (N)
        #   gap     largest |commanded absolute joint target - joint position the command was computed from| over
        #           the 7 joints, in mrad, at every step. Same definition as scripts/isaaclab/demo_z_bands.py,
        #           so the numbers can be compared with the demos' (about 6 mrad inside the hole).
        self._forces = []
        self._gaps = []
        self._last_q = None           # joint positions of the last observation
        self._last_gap = None         # gap of the last command (video banner)

        # Taxels (robot profile with tactile zones: the server then sends extra/tactile, (2, 7, 4) N). The video panel
        # below the cameras is drawn for every robot (a placeholder without taxels) so that all videos have the same
        # 448 x 448 layout; `video_tactile_panel: false` in the YAML turns it off (cameras only, 448 x 224).
        self._tactile = None          # last extra/tactile of the episode
        self._tactile_max = []        # max over the zones, at every observation (N)
        self._tactile_panel = bool(getattr(cfg, "video_tactile_panel", True))
        self._tactile_vmax = float(getattr(cfg, "tactile_vmax", TACTILE_VMAX))

        logging.info(f"IsaacLabEnvWrapper: connected, task={self.task_description!r}")

    def _banner(self, info, verdict, width):
        """Two text lines under a video frame, like the demo collector's videos: step, peg offset
        (xy distance, z, x, y), closest approach so far, the server's "inserted" flag and the verdict
        of the whole episode. Returns a (BANNER_H, width, 3) uint8 array, or None if PIL is missing."""
        mm = info["offset_mm"]
        xy = float(np.hypot(mm[0], mm[1]))
        force, gap, tac = info.get("force_n"), info.get("gap_mrad"), info.get("tactile_max_n")
        line1 = (f"t={info['step']:3d}  xy={xy:6.2f} mm  z={mm[2]:+6.1f} mm   "
                 f"x={mm[0]:+6.2f} y={mm[1]:+6.2f}")
        line2 = (f"closest xy {info['closest_xy_mm']:5.2f} mm @t={info['closest_step']:<3d} | "
                 f"{'INSERTED' if info['flag'] else '-':8s} | episode: {verdict}")
        line3 = (f"force {'-' if force is None else f'{force:5.2f} N'}  |  "
                 f"cmd gap {'-' if gap is None else f'{gap:5.1f} mrad'}"
                 + ("" if tac is None else f"  |  taxel max {tac:5.2f} N"))
        return text_banner([
            (line1, (255, 255, 255)),
            (line2, (90, 220, 120) if (info["flag"] or verdict == "SUCCESS") else (200, 200, 200)),
            (line3, (120, 180, 255))], width, self.BANNER_H)

    def _flush_video(self):
        """Write the accumulated frames of the episode just finished. Called
        from reset() (previous episode) and close() (the last one, which
        never gets a following reset() to trigger it). Each frame carries a banner with the live
        peg-to-socket metrics; the verdict of the episode is only known now, so it is drawn here."""
        if self._video_dir is not None and len(self._frames) > 0:
            path = os.path.join(self._video_dir, f"episode_{self._episode_count}.mp4")
            import imageio.v3 as iio
            verdict = ("SUCCESS" if self._success else "fail") if self._hold_steps > 0 else "n/a"
            frames = self._frames
            if len(self._frame_infos) == len(frames):
                with_banner = []
                for frame, info in zip(frames, self._frame_infos):
                    banner = self._banner(info, verdict, frame.shape[1])
                    with_banner.append(frame if banner is None else np.concatenate([frame, banner], axis=0))
                frames = with_banner
            iio.imwrite(path, frames, fps=10, codec="libx264")
            self._frames = []
            self._frame_infos = []
            self._episode_count += 1

    def _record_metrics(self, obs):
        off = obs.get("extra/peg_socket_offset")
        if off is not None:
            if not self._peg_offsets:                       # first observation of an episode
                self._closest = (float("inf"), 0)
            off = np.asarray(off, dtype=np.float64).reshape(-1)[:3]
            self._peg_offsets.append(off)
            q = obs.get("observation/joint_position")
            self._last_q = None if q is None else np.asarray(q, dtype=np.float64).reshape(-1)[:7]
            tactile = obs.get("extra/tactile")
            self._tactile = None if tactile is None else np.asarray(tactile, dtype=np.float32)
            if self._tactile is not None:
                self._tactile_max.append(tactile_stats(self._tactile)[0])
            force = obs.get("extra/ft_force")
            if force is not None:
                self._forces.append(float(np.linalg.norm(np.asarray(force, dtype=np.float64).reshape(-1)[:3])))
            xy = float(np.hypot(off[0], off[1]))
            if xy < self._closest[0]:
                self._closest = (xy, len(self._peg_offsets) - 1)

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
                  "success needs xy < 2.5 mm AND z below its height threshold. "
                  + (f"An episode counts as a success after {self._hold_steps} such steps in a row."
                     if self._hold_steps > 0 else "success_hold_steps = 0: legacy success flag (always False)."),
                  flush=True)
            self._metrics_header_printed = True
        verdict = ("SUCCESS" if self._success else "fail") if self._hold_steps > 0 else "n/a"
        n_last = self.LAST_N
        extra = {"z_change_last20_mm": mm(o[-1, 2] - o[-min(n_last, len(o)), 2])}
        if self._forces:
            f = np.asarray(self._forces)
            extra["force_last20_N"] = float(f[-n_last:].mean())
            extra["force_max_N"] = float(f.max())
        if self._gaps:
            g = np.asarray(self._gaps)
            extra["gap_last20_mrad"] = float(g[-n_last:].mean())
            extra["gap_mean_mrad"] = float(g.mean())
        if self._tactile_max:                    # robot with taxels: max over the 56 zones, per step (net contact force)
            t = np.asarray(self._tactile_max)
            extra["tactile_max_N"] = float(t.max())
            extra["tactile_last20_N"] = float(t[-n_last:].mean())
        extra_txt = ""
        if "force_last20_N" in extra:
            extra_txt += f" | force last {n_last} steps {extra['force_last20_N']:.2f} N (max {extra['force_max_N']:.2f})"
        if "gap_last20_mrad" in extra:
            extra_txt += f" | command gap last {n_last} steps {extra['gap_last20_mrad']:.1f} mrad"
        if "tactile_max_N" in extra:
            extra_txt += f" | taxel max {extra['tactile_max_N']:.2f} N (last {n_last} steps {extra['tactile_last20_N']:.2f})"
        print(
            f"[peg-socket] episode {self._metrics_episode}: start xy={mm(xy[0]):.1f} mm z={mm(o[0, 2]):+.1f} mm"
            f" | closest xy={mm(xy[k]):.1f} mm (z={mm(o[k, 2]):+.1f} mm) at step {k}"
            f" | final x={mm(o[-1, 0]):+.1f} y={mm(o[-1, 1]):+.1f} mm (xy={mm(xy[-1]):.1f}) z={mm(o[-1, 2]):+.1f} mm"
            f" | {verdict} (longest success streak {self._max_streak}, {len(o)} steps)" + extra_txt,
            flush=True,
        )
        if self._video_dir is not None:
            record = {
                "episode": self._metrics_episode, "steps": int(len(o)),
                "start_xy_mm": mm(xy[0]), "closest_xy_mm": mm(xy[k]), "closest_step": k,
                "closest_z_mm": mm(o[k, 2]), "final_offset_mm": [mm(v) for v in o[-1]],
                "final_xy_mm": mm(xy[-1]),
                "success": bool(self._success) if self._hold_steps > 0 else None,
                "max_success_streak": int(self._max_streak), "success_hold_steps": self._hold_steps,
                **extra,
            }
            with open(os.path.join(self._video_dir, "peg_socket_metrics.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        self._peg_offsets = []
        self._forces = []
        self._gaps = []
        self._tactile_max = []
        self._metrics_episode += 1

    def reset(self, **reset_kwargs):
        """reset_kwargs: currently only `seed` is forwarded to the server —
        see env_client.EnvClient.reset for the wire-protocol side of this."""
        self._flush_video()
        self._flush_metrics()
        obs = self._client.reset(seed=reset_kwargs.get("seed"))
        self._record_metrics(obs)
        self._done = False
        self._server_done = False
        self._success = False
        self._reward = 0.0
        self._streak = 0
        self._max_streak = 0
        self._last_gap = None
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
        if self._last_q is not None:                      # how far the command leads the arm (see __init__)
            self._last_gap = 1000.0 * float(np.max(np.abs(action[:7].astype(np.float64) - self._last_q)))
            self._gaps.append(self._last_gap)
        real_action, action_type = self._client.step(action)
        done, success, reward, _mask = self._client.get_info_for_step()
        self._server_done = bool(done)
        self._flag = bool(success) and not self._server_done     # not the post-reset flag of a timeout step
        if self._hold_steps > 0:
            if done:
                # Timeout: the simulation was reset inside this step, so the flag and the reward belong
                # to the next episode's start state. Not a measurement of this episode.
                success, reward = False, 0.0
            else:
                self._streak = self._streak + 1 if success else 0
                self._max_streak = max(self._max_streak, self._streak)
                held = self._streak >= self._hold_steps
                success, done = held, held        # inserted and held long enough: success, end the episode
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
        if self._server_done:
            # The step that ended the episode has already auto-reset the simulation (Isaac Lab
            # resets inside env.step()): this observation belongs to the NEXT episode. Recording it
            # would put a random start state at the end of the metrics and the video. (An episode
            # ended early by the success rule is NOT in this case: the simulation was not reset.)
            return obs
        self._record_metrics(obs)
        if self._video_dir is not None:
            # sim server emits "observation/..." keys, the real-robot server plain keys
            ext = obs.get("observation/exterior_image_1_left", obs.get("exterior_image_1_left"))
            wrist = obs.get("observation/wrist_image_left", obs.get("wrist_image_left"))
            if ext is not None and wrist is not None:
                if self._tactile_panel:      # cameras side by side, the taxel panel below: 448 x 448
                    tiled = compose_frame(ext, wrist, self._tactile, vmax=self._tactile_vmax)
                else:
                    tiled = np.concatenate([np.asarray(ext, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)], axis=1)
                self._frames.append(tiled)
                off = self._peg_offsets[-1] if self._peg_offsets else np.zeros(3)
                self._frame_infos.append({
                    "step": max(len(self._peg_offsets) - 1, 0),          # control steps done (0 = reset state)
                    "offset_mm": (1000.0 * off).tolist(),
                    "closest_xy_mm": 1000.0 * self._closest[0] if np.isfinite(self._closest[0]) else 0.0,
                    "closest_step": self._closest[1],
                    "flag": self._flag,
                    "force_n": self._forces[-1] if self._forces else None,
                    "gap_mrad": self._last_gap,
                    "tactile_max_n": self._tactile_max[-1] if self._tactile_max else None,
                })
        return obs

    def get_info_for_step(self):
        mask = 1.0 - float(self._done)
        return self._done, self._success, self._reward, mask

    def get_raw_info(self):
        return dict(self._info)

    def close(self):
        self._flush_video()
        self._flush_metrics()
