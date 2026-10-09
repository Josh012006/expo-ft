"""Smoke test for the Isaac Lab / FORGE env server — plumbing only, no model.

Run in .venv, AFTER forge_server.py is up and listening (see the task YAML's server_host /
server_port). No pi05_droid_jointpos involved.

The server's action is 8D: 7 ABSOLUTE joint targets (rad) + a gripper command (0 = open,
1 = closed). An all-zero action would therefore drive every joint to 0 rad: "hold" below means
"command the measured pose", never zeros.

Checks, in order:
  1. actual DOF ordering (joint_names) instead of trusting [0:7]=arm, [7:9]=fingers
  2. reset(seed=...) twice, observation shapes, force/threshold not flat zero
  3. HOLD: commanding the measured pose must not move any joint (position and velocity printed)
  4. gripper command direction (skipped when the YAML sets gripper_can_open: false)
  5. ABSOLUTE-TARGET RAMP on one joint: ramp to start+offset, settle, check the arm sits at the
     commanded ABSOLUTE value, re-hold the same target and check it does not keep moving (an
     increment interpretation would keep going), then ramp back and check the round trip both in
     joint space and in FORGE's own fingertip pose (computed independently of our action code)
  6. frames before / mid / after saved for a visual check

Usage:
    python scripts/isaaclab/smoke_test_env_server.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml
"""
import argparse
import os

import imageio.v3 as iio
import numpy as np

from expo_ft.utils.config_loader import load_task_config
from expo_ft.env.isaaclab_env import IsaacLabEnvWrapper

GRIPPER_OPEN_WIDTH = 0.04  # m per finger (forge_env_patched.GRIPPER_OPEN_WIDTH)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--out", default="smoke_test_frames")
parser.add_argument("--move-joint", type=int, default=1, help="Arm joint index [0-6] for the ramp test.")
parser.add_argument("--offset", type=float, default=0.2, help="Excursion of that joint (rad).")
parser.add_argument("--move-steps", type=int, default=20, help="Steps of the linear ramp, each way.")
parser.add_argument("--settle-steps", type=int, default=40)
parser.add_argument("--tolerance", type=float, default=0.03, help="Accepted error on reaching a target (rad).")
args = parser.parse_args()

cfg = load_task_config(args.config)
env = IsaacLabEnvWrapper(
    {"example_action": np.zeros(8, dtype=np.float32), "env_usage": "eval", "video_dir": None},
    cfg,
)
print(f"connected — env_id={env.env_id!r} task_description={env.task_description!r}")


def debug(kind):
    return env._client.client._call_operation("debug", {"env_id": env.env_id, "kind": kind})["result"]


def hold_action(obs):
    """Command the measured pose: absolute joint targets = current joints, gripper = current."""
    q = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)
    width = float(debug("finger_width"))        # raw metres (the observation itself follows `gripper_obs`)
    cmd = float(np.clip(1.0 - width / GRIPPER_OPEN_WIDTH, 0.0, 1.0))
    return np.concatenate([q, [cmd]]).astype(np.float32)


force_log = []   # |wrist force/torque sensor force| at every step, to tell contact from a controller error


def step(action):
    env.step(action)
    o = env.get_observation()
    force_log.append(float(np.linalg.norm(o["extra/ft_force"])))
    return o


def fingertip():
    return np.array(debug("fingertip_pose")["pos"])


# -- 1. DOF ordering ----------------------------------------------------------------------
joint_names = debug("joint_names")
print(f"\njoint_names (index order): {joint_names}")
assert len(joint_names) == 9, f"expected 9 DOFs (7 arm + 2 fingers), got {len(joint_names)}"
print(f"  -> [0:7] arm:     {joint_names[0:7]}")
print(f"  -> [7:9] fingers: {joint_names[7:9]}")

# -- 2. resets ----------------------------------------------------------------------------
obs = env.reset(seed=0)
print("\nreset(seed=0) observation:")
for k, v in obs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype} min={v.min()} max={v.max()}")
    else:
        print(f"  {k}: {v!r}")

obs2 = env.reset(seed=1)
print(f"\nreset(seed=1) joint_position: {obs2['observation/joint_position']}")
print(f"extra/ft_force: {obs2['extra/ft_force']}   extra/force_threshold: {obs2['extra/force_threshold']}")
if obs2["extra/force_threshold"] == 0.0:
    print("  WARNING: force_threshold is exactly 0.0 — check the contact_penalty_thresholds randomization.")

# -- 3. hold ------------------------------------------------------------------------------
print("\nHOLD (command the measured pose), 60 steps from reset(seed=1):")
hold = hold_action(obs2)
start = np.asarray(obs2["observation/joint_position"], dtype=np.float64)
for t in range(60):
    o = step(hold)
    if t % 10 == 9:
        pos = np.asarray(o["observation/joint_position"], dtype=np.float64)
        vel = np.array(debug("joint_velocity"))
        print(f"  step {t}: max drift {np.abs(pos - start).max():.4f} rad, max |velocity| {np.abs(vel).max():.4f} rad/s")
drift = np.asarray(o["observation/joint_position"], dtype=np.float64) - start
for j in range(7):
    flag = "BUG" if abs(drift[j]) > 0.02 else "ok "
    print(f"  joint[{j}] ({joint_names[j]}): drift={drift[j]:+.4f}  [{flag}]")
print("HOLD:", "PASS" if np.abs(drift).max() <= 0.02 else "FAIL")
rest_force = float(np.median(force_log))
print(f"wrist force at rest (median over the hold): {rest_force:.3f} N")

# -- 4. gripper ---------------------------------------------------------------------------
if getattr(cfg, "gripper_can_open", True):
    obs3 = env.reset(seed=1)
    base = hold_action(obs3)
    widths = {}
    for name, value in (("close(1.0)", 1.0), ("open(0.0)", 0.0)):
        a = base.copy()
        a[7] = value
        for _ in range(10):
            o = step(a)
        widths[name] = float(debug("finger_width"))
    print(f"\ngripper {widths}")
    print("GRIPPER:", "PASS" if abs(widths["open(0.0)"] - widths["close(1.0)"]) > 0.002 else "FAIL (stuck)")
else:
    print("\ngripper test skipped: gripper_can_open is false in the YAML (the gripper channel is ignored).")

# -- 5. absolute-target ramp --------------------------------------------------------------
os.makedirs(args.out, exist_ok=True)
obs = env.reset(seed=1)
j = args.move_joint
q_start = np.asarray(obs["observation/joint_position"], dtype=np.float64)
grip_cmd = hold_action(obs)[7]
tip_start = fingertip()


def command(q_target):
    return np.concatenate([q_target, [grip_cmd]]).astype(np.float32)


def ramp(q_from, q_to):
    for k in range(args.move_steps):
        step(command(q_from + (q_to - q_from) * (k + 1) / args.move_steps))


def settle(q_target, steps):
    last = None
    for _ in range(steps):
        last = step(command(q_target))
    return np.asarray(last["observation/joint_position"], dtype=np.float64), last


def save(tag, o):
    iio.imwrite(f"{args.out}/exterior_{tag}.png", o["observation/exterior_image_1_left"])
    iio.imwrite(f"{args.out}/wrist_{tag}.png", o["observation/wrist_image_left"])


save("before", obs)
force_start = len(force_log)
q_target = q_start.copy()
q_target[j] += args.offset
print(f"\nABSOLUTE RAMP on joint[{j}]: {q_start[j]:+.4f} -> {q_target[j]:+.4f} (offset {args.offset:+.3f} rad)")

ramp(q_start, q_target)
q_mid, o_mid = settle(q_target, args.settle_steps)
save("mid", o_mid)
err_reach = abs(q_mid[j] - q_target[j])
print(f"  after settling at the target: joint[{j}] = {q_mid[j]:+.4f}  (commanded {q_target[j]:+.4f}, error {err_reach:.4f})")
print("  REACHES THE COMMANDED ABSOLUTE VALUE:", "PASS" if err_reach <= args.tolerance else "FAIL")
force_at_target = force_log[-1]
print(f"  wrist force: rest {rest_force:.3f} N, peak during ramp+settle {max(force_log[force_start:]):.3f} N, "
      f"at the target {force_at_target:.3f} N")
if err_reach > args.tolerance:
    print("  -> if the force at the target is far above rest, the arm is PRESSING ON SOMETHING (contact: held peg "
          "on the socket/table), not a controller error. Re-run with the opposite sign (--offset "
          f"{-args.offset:+.2f}) or another joint (--move-joint 0 / 6) to compare.")

q_later, _ = settle(q_target, 20)
creep = abs(q_later[j] - q_mid[j])
print(f"  re-holding the SAME absolute target 20 more steps: joint moved {creep:.4f} rad")
print("  DOES NOT KEEP MOVING (not an increment):", "PASS" if creep <= 0.01 else "FAIL")
tip_mid = fingertip()
print(f"  fingertip (FORGE-native): moved {np.linalg.norm(tip_mid - tip_start):.4f} m from the start")

ramp(q_target, q_start)
q_end, o_end = settle(q_start, args.settle_steps)
save("after", o_end)
tip_end = fingertip()
err_back = float(np.abs(q_end - q_start).max())
tip_back = float(np.linalg.norm(tip_end - tip_start))
print(f"  back at the start pose: max joint error {err_back:.4f} rad, fingertip error {tip_back:.4f} m")
print("  ROUND TRIP (joints):", "PASS" if err_back <= args.tolerance else "FAIL")
print("  ROUND TRIP (fingertip, FORGE-native):", "PASS" if tip_back <= 0.01 else "FAIL")
print(f"\nFrames saved to {args.out}/ (exterior_/wrist_ before, mid, after).")
