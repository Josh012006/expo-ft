"""Isaac Lab / FORGE environment server.

Run inside .venv-isaaclab ONLY (needs isaacsim/isaaclab installed):

    python -m expo_ft.env.isaaclab.forge_server --config configs/task/isaaclab/peg_insert_forge_pi05.yaml

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
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401  (registers Isaac-Forge-*-Direct-v0)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from expo_ft.env.isaaclab.forge_env_patched import ForgeEnvJointPosPi05  # noqa: E402
from expo_ft.env.isaaclab.ws_env_server import WsEnvServer  # noqa: E402


def build_env(cfg_yaml: dict):
    task_name = cfg_yaml["task_name"]  # e.g. "Isaac-Forge-PegInsert-Direct-v0"
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=1)

    # Match pi05_droid_jointpos: 8D (7 joint-position deltas + gripper),
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
        env_cfg.robot.actuators[group].damping = 4.0

    # One render per env step (env step = decimation physics steps).
    env_cfg.sim.render_interval = env_cfg.decimation

    if not cfg_yaml.get("realtime_throttle", False):
        pass  # default: run as fast as the GPU allows (see task YAML comment)

    env = ForgeEnvJointPosPi05(cfg=env_cfg, render_mode=None)
    return env, task_name


class ForgeBackend:
    """Implements ws_env_server.Backend for a single ForgeEnvJointPosPi05 instance."""

    def __init__(self, cfg_yaml: dict):
        self.cfg_yaml = cfg_yaml
        self.env, self.task_name = build_env(cfg_yaml)
        self.prompt = cfg_yaml.get("language_instruction", "")
        self._env_id = "isaaclab_forge_0"
        self._zero_action = torch.zeros((1, self.env.cfg.action_space), device=self.env.device)
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
        return {"error": f"unknown debug kind: {kind}"}

    def reset(self, env_id: str, seed):
        # Gymnasium/Isaac Lab standard reset(seed=...) contract. Confirm on
        # the first live run that this actually re-seeds FORGE's fixed-asset
        # randomization (not yet verified end-to-end) — if not, fall back to
        # torch.manual_seed(seed) before calling reset().
        if seed is not None:
            self.env.reset(seed=int(seed))
        else:
            self.env.reset()
        self.env.step(self._zero_action)  # let camera/observation buffers populate
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

    def get_observation(self, env_id: str) -> dict:
        env = self.env
        ext = env._exterior_cam.data.output["rgb"][0].cpu().numpy()
        wrist = env._wrist_cam.data.output["rgb"][0].cpu().numpy()
        joint_pos = env.joint_pos[0, 0:7].detach().cpu().numpy()
        gripper_pos = env.joint_pos[0, 7:8].detach().cpu().numpy()
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
