"""Isaac Lab / FORGE environment server.

Run inside .venv-isaaclab ONLY (needs isaacsim/isaaclab installed):

    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -m expo_ft.env.isaaclab.forge_server --config configs/task/isaaclab/peg_insert_forge_pi05.yaml

The CUBLAS_WORKSPACE_CONFIG export is required, not optional, and must be set
BEFORE this process starts — CUDA initializes as soon as Isaac Sim/AppLauncher
comes up, and PyTorch's own deterministic-algorithms setting (forced by
ForgeBackend.reset() for reproducible reset(seed=...) results, see below) has
no effect on CuBLAS ops without it: reset() raises a RuntimeError instead.
Confirmed in testing: with this set, reset(seed=N) now reproduces the exact
same joint_position bit-for-bit across separate server runs; without it,
observed drift up to ~0.03 rad on panda_joint1 between runs with the same seed.

This process owns the physical world (the simulation) and is the SERVER in
this project's client-server convention — the same role the real DROID
robot's controller plays for expo_ft.env.env_client.EnvClient. The caller
(train_pi_robo.py / eval scripts, running pi05_droid_jointpos) is the client.
That choice, and why it's the opposite of Isaac Lab Arena's own
model-is-the-server convention, is documented in the task YAML and in the
design-decisions summary — not repeated here.

One environment only (num_envs=1); the task is fixed for the lifetime of
this process, read from --config — reloading a different USD scene mid-run
would cost the ~2 minute Isaac Sim startup penalty again, so `create_env` is
a formality here (there is only ever one instance to hand out), not a
task switch.
"""
import argparse
import logging
import os
import signal
import sys

import yaml
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to the task YAML (see configs/task/isaaclab/).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

with open(args.config) as f:
    cfg_yaml = yaml.safe_load(f)

args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

# AppLauncher installs its own SIGINT handler that attempts a Kit-internal
# shutdown sequence and, observed in testing, never actually returns —
# Ctrl+C did nothing and the GPU stayed allocated. Defining the handler here
# (needs `app` in scope) but NOT registering it yet — see main(), where it's
# installed as the LAST thing before serve_forever(). Registering it here
# instead didn't survive: Kit/Isaac Sim scene construction (ForgeEnvJointPosPi05
# below) apparently re-installs its own handler sometime after AppLauncher
# but before the env is fully built, silently overwriting ours again.
_backend_holder = {"backend": None}


def _handle_shutdown_signal(signum, frame):
    # print(..., flush=True) as well as logging: if Kit has redirected/buffered
    # Python's logging output, this is a second, harder-to-swallow channel to
    # confirm the handler actually fired at all versus hanging in cleanup.
    print(f"[forge_server] received {signal.Signals(signum).name} — shutting down.", flush=True)
    logging.info("Received signal %s — shutting down.", signal.Signals(signum).name)
    backend = _backend_holder["backend"]
    try:
        if backend is not None:
            backend.env.close()
        app.close()
    except Exception:
        logging.exception("Error during shutdown cleanup (continuing to exit anyway):")
    os._exit(0)


# Everything Isaac-Sim-dependent is imported only after AppLauncher exists.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401  (registers Isaac-Forge-*-Direct-v0)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from expo_ft.env.isaaclab.forge_env_patched import ForgeEnvJointPosPi05, GRIPPER_OPEN_WIDTH  # noqa: E402
from expo_ft.env.isaaclab.ws_env_server import WsEnvServer  # noqa: E402


GRIPPER_OBS_MODES = ("raw", "normalized", "closed")


def gripper_obs_value(width, mode):
    """Gripper observation sent to the policy, from the finger joint position `width` (m).

    FORGE: 0.0 = fingers closed, GRIPPER_OPEN_WIDTH (0.04 m, Franka finger travel) = fully open.
    pi05_droid_jointpos expects the DROID convention: 0 = open, 1 = closed.
      raw        : the width in metres, unchanged (legacy behaviour, not the model's convention)
      normalized : 1 - width / GRIPPER_OPEN_WIDTH, in [0, 1] -- the exact inverse of the mapping
                   ForgeEnvJointPosPi05 applies to the gripper ACTION. With the peg held the
                   fingers rest on it, so this reads ~0.9 (peg half-width 4 mm), not 1.0.
      closed     : constant 1.0 (use only when the gripper is held closed for the whole episode)
    """
    if mode == "raw":
        return width
    if mode == "normalized":
        return np.clip(1.0 - width / GRIPPER_OPEN_WIDTH, 0.0, 1.0)
    return np.ones_like(width)


def build_env(cfg_yaml: dict):
    task_name = cfg_yaml["task_name"]  # e.g. "Isaac-Forge-PegInsert-Direct-v0"
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=1)

    # Match pi05_droid_jointpos: 8D (7 absolute joint-position targets + gripper),
    # replacing FORGE's native 7D task-space + success-prediction action.
    env_cfg.action_space = 8

    # Re-enable PhysX's implicit PD on the arm joints. FORGE zeroes these
    # (stiffness=damping=0.0) to give its own task-space controller full
    # authority over torques; since that controller is bypassed entirely by
    # ForgeEnvJointPosPi05._apply_action, the zeroed gains would otherwise
    # leave the arm with nothing to track a position target with. Values are
    # Isaac Lab's own defaults for this robot (isaaclab_assets FRANKA_PANDA_CFG).
    for group in ("panda_arm1", "panda_arm2"):
        env_cfg.robot.actuators[group].stiffness = 80.0
        # damping=4 is Isaac Lab's own FRANKA_PANDA_CFG default; too soft to
        # track a deliberate joint move precisely (round-trip test
        # error ~0.09 rad at this value). Bumped to 40 for tracking accuracy
        # only — the earlier drift that first motivated this change turned
        # out to be an unrelated bug (a stale set_joint_effort_target() left
        # standing by FORGE's own reset-time close_gripper_in_place() loop,
        # fixed in _apply_action below), confirmed fixed independently of
        # this gain. This value is purely a tracking-quality choice now, no
        # longer tangled with that bug.
        env_cfg.robot.actuators[group].damping = 40.0

    # One render per env step (env step = decimation physics steps).
    env_cfg.sim.render_interval = env_cfg.decimation

    # FORGE's own native episode_length_s (10.0 for PegInsert) truncates
    # episodes at 150 steps @ 15Hz, enforced inside Isaac Lab's own
    # DirectRLEnv base class (done=truncated). One field, max_steps_per_episode,
    # drives BOTH this server-side truncation and the client eval loop's own
    # cap (scripts/eval_policy.py's `while ... steps < cfg.max_steps_per_episode`)
    # — not two separate settings: there used to be a second, server-only
    # field (max_episode_length) which just invited the two to drift apart
    # silently. Converted to seconds here since that's the unit Isaac Lab's
    # own cfg field expects, using env_cfg's own decimation/physics dt rather
    # than a hardcoded 15 Hz, so this stays correct if either ever changes.
    if "max_steps_per_episode" in cfg_yaml:
        dt_per_step = env_cfg.decimation * env_cfg.sim.dt
        env_cfg.episode_length_s = cfg_yaml["max_steps_per_episode"] * dt_per_step

    if not cfg_yaml.get("realtime_throttle", False):
        pass  # default: run as fast as the GPU allows (see task YAML comment)

    env = ForgeEnvJointPosPi05(
        cfg=env_cfg, render_mode=None,
        gripper_can_open=cfg_yaml.get("gripper_can_open", True),
    )
    return env, task_name


class ForgeBackend:
    """Implements ws_env_server.Backend for a single ForgeEnvJointPosPi05 instance."""

    def __init__(self, cfg_yaml: dict):
        self.cfg_yaml = cfg_yaml
        self.env, self.task_name = build_env(cfg_yaml)
        self.prompt = cfg_yaml.get("language_instruction", "")
        self.gripper_obs = cfg_yaml.get("gripper_obs", "raw")
        if self.gripper_obs not in GRIPPER_OBS_MODES:
            raise ValueError(f"gripper_obs must be one of {GRIPPER_OBS_MODES}, got {self.gripper_obs!r}")
        self._env_id = "isaaclab_forge_0"
        self._last_done = False
        self._last_success = False
        self._last_reward = 0.0

    # -- Backend protocol -------------------------------------------------

    def create_env(self, request: dict):
        return self._env_id, self.prompt

    def get_joint_names(self):
        """Diagnostic only, not part of the Backend protocol proper — lets the
        smoke test verify the [0:7]=arm, [7:9]=fingers indexing assumption
        against this USD's actual DOF order instead of trusting it blind."""
        return list(self.env._robot.data.joint_names)

    def debug(self, request: dict):
        """Dispatched by ws_env_server's optional "debug" operation."""
        kind = request.get("kind")
        if kind == "joint_names":
            return self.get_joint_names()
        if kind == "joint_velocity":
            return self.env.joint_vel[0, 0:7].detach().cpu().numpy().tolist()
        if kind == "fingertip_pose":
            # Ground truth INDEPENDENT of our _apply_action: computed by
            # FORGE's own _compute_intermediate_values (inherited, unmodified)
            # from the robot's actual simulated rigid-body pose, not from
            # anything our joint-target code writes. Used to check that a
            # joint-space target produces sensible, smooth EEF-space motion,
            # not just a round trip in joint-angle terms.
            pos = self.env.fingertip_midpoint_pos[0].detach().cpu().numpy().tolist()
            quat = self.env.fingertip_midpoint_quat[0].detach().cpu().numpy().tolist()
            return {"pos": pos, "quat": quat}
        # -- Used by scripts/isaaclab/collect_waypoint_demos.py (scripted demos) --------------
        # Simulation-only helpers: a real robot has no such op. They only READ the state of the
        # env and return numbers; nothing here moves the arm (the client does that via "step").
        if kind == "finger_width":      # raw finger joint position (m), whatever `gripper_obs` is
            return float(self.env.joint_pos[0, 7])
        if kind == "task_geometry":
            return self._task_geometry()
        if kind == "ik_joint_target":
            return self._ik_joint_target(request["delta_pos"], request["target_quat"])
        return {"error": f"unknown debug kind: {kind}"}

    def _task_geometry(self):
        """Task constants the waypoint script needs, read from FORGE's own task config so the
        client never duplicates them."""
        task = self.env.cfg_task
        hole_depth = float(task.fixed_asset_cfg.height)   # 0.025 m for the 8 mm PegInsert
        return {
            "hole_depth": hole_depth,
            # xy tolerance is hard-coded in FactoryEnv._get_curr_successes (0.0025 m).
            "success_xy": 0.0025,
            # Same formula as _get_curr_successes for peg_insert: fraction of the hole depth.
            "success_z": hole_depth * float(task.success_threshold),
        }

    def _ik_joint_target(self, delta_pos, target_quat):
        """ONE inverse-kinematics step: the 7 ABSOLUTE joint targets that move the fingertip by
        `delta_pos` (m, env frame) while turning it towards `target_quat` (w, x, y, z).

        Uses FORGE's own IK helpers (factory_control.get_pose_error / get_delta_dof_pos, damped
        least squares), the same ones its set_pos_inverse_kinematics() uses at reset, applied to
        the Jacobian and joint state the env has already computed after the last step. Returns
        q_now + dq, i.e. exactly what we would send as the arm part of a "step" action.
        """
        from isaaclab_tasks.direct.factory import factory_control

        env = self.env
        delta = torch.tensor(delta_pos, dtype=torch.float32, device=env.device).reshape(1, 3)
        quat = torch.tensor(target_quat, dtype=torch.float32, device=env.device).reshape(1, 4)

        pos_error, rot_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=env.fingertip_midpoint_pos,
            fingertip_midpoint_quat=env.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=env.fingertip_midpoint_pos + delta,
            ctrl_target_fingertip_midpoint_quat=quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        dq = factory_control.get_delta_dof_pos(
            delta_pose=torch.cat((pos_error, rot_error), dim=-1),
            ik_method="dls",
            jacobian=env.fingertip_midpoint_jacobian,
            device=env.device,
        )
        q_target = env.joint_pos[:, 0:7] + dq[:, 0:7]
        return q_target[0].detach().cpu().numpy().tolist()

    def _hold_action(self):
        """Action that keeps the arm and the gripper where they are. Actions are ABSOLUTE joint
        targets, so an all-zero action would drive every joint to 0 rad."""
        data = self.env._robot.data
        q = data.joint_pos[0, 0:7]
        cmd = (1.0 - data.joint_pos[0, 7] / GRIPPER_OPEN_WIDTH).clamp(0.0, 1.0)   # 0 open, 1 closed
        return torch.cat([q, cmd.reshape(1)]).reshape(1, -1).to(self.env.device)

    def reset(self, env_id: str, seed):
        # env.reset(seed=...) internally calls configure_seed(seed) with
        # torch_deterministic left at its default False (isaaclab/utils/seed.py),
        # which explicitly sets cudnn.benchmark=True / cudnn.deterministic=False
        # — non-deterministic GPU kernels are allowed even with a fixed seed.
        # The random *draws* are reproducible; the physics computation itself
        # (in particular the 0.25s grasp-settling loop inside _reset_idx,
        # which involves GPU contact resolution) isn't guaranteed to be, and
        # testing confirmed joint_position after reset(seed=1) drifting by
        # ~0.01-0.03 rad on panda_joint1 across repeated calls with the same
        # seed. Calling configure_seed ourselves with torch_deterministic=True
        # BEFORE env.reset() (and not passing seed to env.reset() itself, to
        # avoid a second, non-deterministic reseed overwriting this one) is
        # the fix being tested for that.
        if seed is not None:
            from isaaclab.utils.seed import configure_seed
            configure_seed(int(seed), torch_deterministic=True)
            self.env.reset()
        else:
            self.env.reset()
        self.env.step(self._hold_action())  # let camera/observation buffers populate
        obs = self.get_observation(env_id)
        self._last_done = False
        return obs, False

    def step(self, env_id: str, action):
        # torch.tensor() (not as_tensor) — msgpack_numpy unpacks into
        # read-only arrays; as_tensor's zero-copy share triggered a
        # writable-tensor-from-read-only-array warning.
        action_t = torch.tensor(action, dtype=torch.float32, device=self.env.device).reshape(1, -1)
        _, _dense_reward_unused, terminated, truncated, _ = self.env.step(action_t)
        self._last_done = bool(terminated.item() or truncated.item())
        # Sparse, ground-truth reward = success indicator — see design-decisions
        # summary for why FORGE's own dense _get_rewards() (discarded above,
        # it's what produced the smoothly-decaying ~0.29->0.23 values seen in
        # testing) isn't used: it depends on the 7th native-action dimension
        # we don't have.
        self._last_success = bool(
            self.env._get_curr_successes(
                success_threshold=self.env.cfg_task.success_threshold, check_rot=False
            )[0].item()
        )
        self._last_reward = 1.0 if self._last_success else 0.0
        return action, "policy"

    def _peg_socket_offset(self):
        """(x, y, z) in metres, env frame: held peg base minus its TARGET pose in the socket.
        Same quantities FORGE's own _get_curr_successes() uses (factory_utils helpers), computed
        from fresh asset poses: xy = norm(offset[:2]) is FORGE's `xy_dist` (success needs < 2.5 mm)
        and z = its `z_disp` (success needs it below a height threshold). Diagnostic only."""
        from isaaclab_tasks.direct.factory import factory_utils
        env = self.env
        held_pos = env._held_asset.data.root_pos_w - env.scene.env_origins
        fixed_pos = env._fixed_asset.data.root_pos_w - env.scene.env_origins
        held_base_pos, _ = factory_utils.get_held_base_pose(
            held_pos, env._held_asset.data.root_quat_w, env.cfg_task.name,
            env.cfg_task.fixed_asset_cfg, env.num_envs, env.device,
        )
        target_pos, _ = factory_utils.get_target_held_base_pose(
            fixed_pos, env._fixed_asset.data.root_quat_w, env.cfg_task.name,
            env.cfg_task.fixed_asset_cfg, env.num_envs, env.device,
        )
        return (held_base_pos - target_pos)[0].detach().cpu().numpy()

    def get_observation(self, env_id: str) -> dict:
        env = self.env
        ext = env._exterior_cam.data.output["rgb"][0].cpu().numpy()
        wrist = env._wrist_cam.data.output["rgb"][0].cpu().numpy()
        joint_pos = env.joint_pos[0, 0:7].detach().cpu().numpy()
        gripper_pos = gripper_obs_value(env.joint_pos[0, 7:8].detach().cpu().numpy(), self.gripper_obs)
        return {
            "observation/exterior_image_1_left": ext.astype("uint8"),
            "observation/wrist_image_left": wrist.astype("uint8"),
            "observation/joint_position": joint_pos.astype("float32"),
            "observation/gripper_position": gripper_pos.astype("float32"),
            "prompt": self.prompt,
            # Not consumed by pi05_droid_jointpos — logged for tactile-phase
            # diagnostics and future use. force_threshold follows FORGE's own
            # native randomization (per design decision), not a fixed value.
            # Direct attribute access (no getattr default): these names come
            # straight from forge_env.py's own _get_observations(), so a
            # wrong name here should crash loudly, not silently return zeros
            # (which is exactly what the previous getattr(..., "ft_force",
            # ...) / getattr(..., "force_threshold", ...) version did — those
            # attributes don't exist on the env at all).
            "extra/ft_force": env.force_sensor_smooth[0, 0:3].detach().cpu().numpy().astype("float32"),
            "extra/force_threshold": float(env.contact_penalty_thresholds[0].item()),
            # Per-step peg-to-socket offset (see _peg_socket_offset); the client wrapper
            # turns it into a per-episode summary.
            "extra/peg_socket_offset": self._peg_socket_offset().astype("float32"),
        }

    def get_info_for_step(self, env_id: str):
        mask = 1.0 - float(self._last_done)
        return self._last_done, self._last_success, self._last_reward, mask


def main():
    backend = ForgeBackend(cfg_yaml)
    _backend_holder["backend"] = backend  # so _handle_shutdown_signal can reach it
    server = WsEnvServer(
        backend,
        host=cfg_yaml.get("server_host", "0.0.0.0"),
        port=cfg_yaml.get("server_port", 8102),
    )
    # Registered LAST, after ForgeBackend/ForgeEnvJointPosPi05 has finished
    # building the scene — see the comment above _handle_shutdown_signal for
    # why registering this any earlier didn't survive.
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    print("[forge_server] shutdown handler armed — Ctrl+C should now work.", flush=True)
    server.serve_forever()  # returns only via _handle_shutdown_signal's os._exit(0)


if __name__ == "__main__":
    main()
