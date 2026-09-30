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
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.env.device).reshape(1, -1)
        _, reward, terminated, truncated, _ = self.env.step(action_t)
        self._last_done = bool(terminated.item() or truncated.item())
        self._last_reward = float(reward.item())
        # Sparse, ground-truth success — see design-decisions summary for why
        # FORGE's own dense _get_rewards() isn't used (depends on the 7th
        # native-action dimension we don't have).
        self._last_success = bool(
            self.env._get_curr_successes(
                success_threshold=self.env.cfg_task.success_threshold, check_rot=False
            )[0].item()
        )
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
            "extra/ft_force": getattr(env, "ft_force", torch.zeros(1, 3))[0].detach().cpu().numpy().astype("float32"),
            "extra/force_threshold": float(getattr(env, "force_threshold", torch.zeros(1))[0].item()),
        }

    def get_info_for_step(self, env_id: str):
        mask = 1.0 - float(self._last_done)
        return self._last_done, self._last_success, self._last_reward, mask


def main():
    backend = ForgeBackend(cfg_yaml)
    server = WsEnvServer(
        backend,
        host=cfg_yaml.get("server_host", "0.0.0.0"),
        port=cfg_yaml.get("server_port", 8102),
    )
    try:
        server.serve_forever()
    finally:
        backend.env.close()
        app.close()


if __name__ == "__main__":
    main()
