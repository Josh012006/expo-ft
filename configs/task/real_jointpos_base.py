"""Real Franka (DROID) driven by pi05_droid_jointpos: 8D joint-position actions.

Same role as real_base.py, for tasks where the policy outputs 7 joint targets + gripper.
Task configs (see pick_jointpos.py) copy these keys on top of their own task settings.
"""
import numpy as np

from configs.task import real_base

# Keys a task config must take from this base (everything else stays task-specific).
JOINT_KEYS = (
    "action_space", "gripper_action_space", "control_hz",
    "max_joint_step", "joint_limit_margin",
    "invert_gripper_observation", "human_override",
    "example_action", "state_obs_key", "state_obs_dim", "output_action_dim",
    "use_cartesian_state", "chunk_action_reference",
)


def get_config():
    config = real_base.get_config()

    # DROID RobotEnv: "joint_position" => DoF 8 (7 joints + gripper), absolute joint targets,
    # executed by DROID through its cartesian-impedance joint-target interface.
    config.action_space = "joint_position"
    config.gripper_action_space = "position"       # command: 0 = open, 1 = closed
    config.control_hz = 15                          # pi05_droid_jointpos nominal rate

    # The 7 action values arriving at DroidEnv.step() are ABSOLUTE joint targets (the client adds
    # the plan-time joint state back onto the model's chunk offsets, see chunk_action_reference).
    # Safety, applied in DroidEnv._joint_step: max |target - measured| per control step (rad).
    # 0.03 rad/step = 0.45 rad/s at 15 Hz: deliberately slow for the first robot tests.
    config.max_joint_step = 0.03
    config.joint_limit_margin = 0.05

    # robot_state gripper is width/max_width (1 = open); the model expects 0 = open.
    config.invert_gripper_observation = True

    # The SpaceMouse override is a 7D cartesian-velocity action: incompatible with 8D joint actions.
    config.human_override = False

    # Shapes / keys seen by the policy client (real server sends un-prefixed keys: the openpi
    # repack transform maps "joint_position" -> "observation/joint_position").
    config.example_action = np.zeros((1, 8))
    config.use_cartesian_state = False
    config.state_obs_key = "joint_position"
    config.state_obs_dim = 7
    config.output_action_dim = 8

    # Client-side: add the joint state at plan time back onto the chunk (eval_policy.py).
    config.chunk_action_reference = "chunk_start_state"
    return config
