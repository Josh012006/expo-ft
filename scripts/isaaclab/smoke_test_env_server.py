"""Smoke test for the Isaac Lab / FORGE env server — plumbing only, no model.

Run in .venv, AFTER forge_server.py is up and listening (see the task YAML's
server_host/server_port). No pi05_droid_jointpos involved: this only tests
that the wire protocol, the joint-position-delta action interface, and the
observation dict are actually correct — three things a zero-action test
cannot distinguish from "nothing is connected to anything".

Checks, in order:
  1. connect + reset (twice, to catch state that doesn't survive a second
     episode in the same server process)
  2. actual DOF ordering (joint_names), instead of trusting the [0:7]=arm,
     [7:9]=fingers assumption blind
  3. a deliberate, nonzero action on ONE arm joint, moved for several steps,
     checking the observed joint_position actually shifted the right way by
     roughly the right amount — this is the first real exercise of
     ForgeEnvJointPosPi05._apply_action; nothing before this script tested
     it with anything but zeros
  4. extra/ft_force and extra/force_threshold are no longer flat zero
     (catches the getattr(..., default) bug that silently masked a wrong
     attribute name)
  5. saves one exterior + one wrist frame before/after the nonzero action,
     for a visual sanity check that the arm actually moved

Usage:
    python scripts/isaaclab/smoke_test_env_server.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml
"""
import argparse
import os

import imageio.v3 as iio
import numpy as np

from expo_ft.utils.config_loader import load_task_config
from expo_ft.env.isaaclab_env import IsaacLabEnvWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--out", default="smoke_test_frames")
parser.add_argument("--move-joint", type=int, default=1, help="Arm joint index [0-6] to move.")
parser.add_argument("--delta", type=float, default=0.02, help="Per-step joint-position delta (rad).")
parser.add_argument("--move-steps", type=int, default=15)
args = parser.parse_args()

cfg = load_task_config(args.config)

env = IsaacLabEnvWrapper(
    {"example_action": np.zeros(8, dtype=np.float32), "env_usage": "eval", "video_dir": None},
    cfg,
)
print(f"connected — env_id={env.env_id!r} task_description={env.task_description!r}")

# -- 2. Real DOF ordering, not assumed -------------------------------------
joint_names = env._client.client._call_operation(
    "debug", {"env_id": env.env_id, "kind": "joint_names"}
)["result"]
print(f"\njoint_names (index order): {joint_names}")
assert len(joint_names) == 9, f"expected 9 DOFs (7 arm + 2 fingers), got {len(joint_names)}"
print(f"  -> indices [0:7] assumed = arm:      {joint_names[0:7]}")
print(f"  -> indices [7:9] assumed = fingers:   {joint_names[7:9]}")
print("  Confirm by eye these are actually panda_joint1..7 then finger joints, in that order.")

# -- 1. reset, twice ---------------------------------------------------------
obs = env.reset(seed=0)
print("\nreset(seed=0) #1 observation:")
for k, v in obs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype} min={v.min()} max={v.max()}")
    else:
        print(f"  {k}: {v!r}")

obs2 = env.reset(seed=1)
print(f"\nreset(seed=1) #2 — joint_position: {obs2['observation/joint_position']}")
print("  (different from reset #1 above if FORGE's fixed-asset randomization "
      "is actually seeded, which is itself worth checking)")

# -- 1b. pure hold test: ALL-ZERO actions, no round trip, no gripper test in
# between — isolates whether any joint drifts with zero commanded delta.
# Motivated by a real anomaly: joint[0]'s reading drifted ~-0.47 rad between
# reset(seed=1) and the start of the round-trip test in an earlier run, across
# phases (gripper test) that sent delta=0 for joint 0 explicitly.
# If _apply_action's hold behavior (target = current + 0) is correct, every
# joint should stay flat here; a joint that drifts while never commanded is a
# real bug, not a contact or gravity effect.
print("\npure hold (all-zero action) drift check, 60 steps from reset(seed=1):")
hold_only = np.zeros(8, dtype=np.float32)
hold_start = obs2["observation/joint_position"].copy()
for t in range(60):
    env.step(hold_only)
    if t % 10 == 9:
        pos = env.get_observation()["observation/joint_position"]
        vel = env._client.client._call_operation(
            "debug", {"env_id": env.env_id, "kind": "joint_velocity"}
        )["result"]
        drift = pos - hold_start
        print(f"  step {t}: joint_position={np.round(pos, 4)}  drift={np.round(drift, 4)}")
        print(f"           joint_velocity={np.round(vel, 4)}  "
              f"(near-zero here + still-changing position above = real bug, not settling)")
hold_end = env.get_observation()["observation/joint_position"]
hold_drift = hold_end - hold_start
print(f"total drift over 60 held steps: {np.round(hold_drift, 4)}")
print("per-joint verdict (threshold 0.02 rad, zero delta commanded the whole time):")
any_bug = False
for j in range(7):
    flag = "BUG" if abs(hold_drift[j]) > 0.02 else "ok "
    if flag == "BUG":
        any_bug = True
    print(f"  joint[{j}] ({joint_names[j]}): drift={hold_drift[j]:+.4f}  [{flag}]")
if any_bug:
    print("\nAt least one joint drifted with zero commanded delta — this is not "
          "contact or gravity, something is setting a nonzero target for that "
          "joint even when action[joint]==0. Check ctrl_target_joint_pos "
          "handling in forge_env_patched.py._pre_physics_step for that index.")
else:
    print("\nPASS — no joint drifts meaningfully under an all-zero action. If this "
          "differs from the earlier round-trip run's baseline anomaly, the drift "
          "there was likely caused by the gripper-test phase specifically, not "
          "a standing bug — worth re-checking with those steps included instead.")



# -- 4. force/threshold not flat zero ---------------------------------------
print(f"\nextra/ft_force: {obs2['extra/ft_force']}")
print(f"extra/force_threshold: {obs2['extra/force_threshold']}")
if obs2["extra/force_threshold"] == 0.0:
    print("  WARNING: still exactly 0.0 — check contact_penalty_thresholds randomization range in the task cfg.")

# -- 3b. gripper close/open, isolated from the arm round trip ---------------
# gripper_position was seen stuck at ~0.0053 during the arm-only round trip
# above, with gripper_cmd=0 (open) sent the whole time. That number is very
# close to FORGE's own reset-time grip width for the peg
# (diameter/2 * 1.25 ~= 0.005-0.0051 for PegInsert's peg, per
# factory_env.py) — i.e. the peg is physically wedged between the fingers,
# so an "open" command pushes against that contact and can't win. Not
# necessarily a bug in the gripper control path itself, just never
# exercised without something in the way. Command CLOSE (1.0) then OPEN
# (0.0) explicitly and check the two settle at genuinely different values —
# that's what actually tells the control path apart from "always stuck".
close_action = np.zeros(8, dtype=np.float32)
close_action[7] = 1.0
open_action = np.zeros(8, dtype=np.float32)
open_action[7] = 0.0

for _ in range(10):
    env.step(close_action)
gripper_closed = env.get_observation()["observation/gripper_position"][0]

for _ in range(10):
    env.step(open_action)
gripper_open = env.get_observation()["observation/gripper_position"][0]

print(f"\ngripper close(1.0) -> {gripper_closed:.4f}   open(0.0) -> {gripper_open:.4f}")
if abs(gripper_open - gripper_closed) > 0.002:
    print("PASS — gripper responds to the command (values genuinely differ); "
          "the peg-contact explanation above is consistent with this.")
else:
    print("Still stuck across an explicit close/open toggle — that IS a real "
          "bug in the gripper control path (mapping sign/scale, or the "
          "actuator not responding at all), not just peg contact. Check "
          "GRIPPER_OPEN_WIDTH / the (1.0 - gripper_cmd) mapping in "
          "forge_env_patched.py._apply_action.")

# -- 3 & 5. deliberate nonzero action on one joint, then its exact opposite -
# Reset first (same seed as obs2, so the starting joint_position is directly
# comparable) — isolates this from the gripper open/close test above, whose
# side effects on the held peg (moved or dropped) would otherwise contaminate
# the arm's contact conditions here and confound the result.
obs3 = env.reset(seed=1)

# The forward-only drift check above (drift ~= delta * steps) can't actually
# tell a correct delta ("target = current_joint_pos + action") from a bug
# that applies the action as an ABSOLUTE target ("target = action") instead:
# both would move the joint by a plausible-looking amount on the first call.
# The decisive test is a round trip: +delta for N steps, then -delta for N
# steps. Under correct delta semantics this returns near the starting
# position, whatever that position is. Under the absolute-target bug, the
# forward phase converges toward the constant +delta and the backward phase
# toward -delta — both small numbers near zero, unrelated to start_pos
# (which is a real joint angle, e.g. the -2.10..2.43 rad range already seen
# in this env, essentially never near zero) — so that bug fails this check
# clearly rather than looking approximately plausible.
os.makedirs(args.out, exist_ok=True)
start_pos = obs3["observation/joint_position"][args.move_joint]


def get_fingertip_pose():
    """FORGE-native ground truth — computed by FORGE's own unmodified
    _compute_intermediate_values from the robot's actual simulated pose, not
    by anything our joint-delta _apply_action writes. Independent check that
    the round trip also makes sense in EEF space, not just in joint angles."""
    r = env._client.client._call_operation(
        "debug", {"env_id": env.env_id, "kind": "fingertip_pose"}
    )["result"]
    return np.array(r["pos"]), np.array(r["quat"])


fingertip_pos_start, _ = get_fingertip_pose()
print(f"\nfingertip_pos (FORGE-native, before round trip): {np.round(fingertip_pos_start, 4)}")

move_fwd = np.zeros(8, dtype=np.float32)
move_fwd[args.move_joint] = args.delta
move_bwd = np.zeros(8, dtype=np.float32)
move_bwd[args.move_joint] = -args.delta

for name, frame_tag in (("exterior_image_1_left", "exterior"), ("wrist_image_left", "wrist")):
    iio.imwrite(f"{args.out}/{frame_tag}_before.png", obs3[f"observation/{name}"])

last_obs = None
for t in range(args.move_steps):
    env.step(move_fwd)
    last_obs = env.get_observation()
    if t in (0, args.move_steps - 1):
        pos = last_obs["observation/joint_position"][args.move_joint]
        print(f"forward step {t}: joint[{args.move_joint}] = {pos:.4f}  (start was {start_pos:.4f}, "
              f"expected drift ~{args.delta * (t + 1):+.4f})")

mid_pos = last_obs["observation/joint_position"][args.move_joint]
fingertip_pos_mid, _ = get_fingertip_pose()
print(f"fingertip_pos (FORGE-native, at mid): {np.round(fingertip_pos_mid, 4)}  "
      f"(moved {np.linalg.norm(fingertip_pos_mid - fingertip_pos_start):.4f} m from start)")
for name, frame_tag in (("exterior_image_1_left", "exterior"), ("wrist_image_left", "wrist")):
    iio.imwrite(f"{args.out}/{frame_tag}_mid.png", last_obs[f"observation/{name}"])

for t in range(args.move_steps):
    env.step(move_bwd)
    last_obs = env.get_observation()
    if t in (0, args.move_steps - 1):
        pos = last_obs["observation/joint_position"][args.move_joint]
        print(f"backward step {t}: joint[{args.move_joint}] = {pos:.4f}")

# Settle: hold position (zero delta) for a few extra steps before reading
# end_pos. Gains are deliberately soft (80/4); reading the position right on
# the last backward step risks measuring a transient still catching up
# rather than where the controller actually settles. This does NOT apply
# any further motion, only lets PD tracking finish converging.
settle_steps = max(5, args.move_steps // 3)
hold_action = np.zeros(8, dtype=np.float32)
for _ in range(settle_steps):
    env.step(hold_action)
    last_obs = env.get_observation()
print(f"settled joint[{args.move_joint}] after {settle_steps} hold steps: "
      f"{last_obs['observation/joint_position'][args.move_joint]:.4f}")

fingertip_pos_end, _ = get_fingertip_pose()
fingertip_round_trip_error = np.linalg.norm(fingertip_pos_end - fingertip_pos_start)
print(f"fingertip_pos (FORGE-native, at end): {np.round(fingertip_pos_end, 4)}  "
      f"(round-trip error in EEF space: {fingertip_round_trip_error:.4f} m)")
if fingertip_round_trip_error > 0.03:
    print("FAIL — the gripper did not return near its starting position in real "
          "3D space (FORGE's own fingertip_midpoint_pos), even if the joint "
          "angle looked fine. This would mean the joint-angle round trip is "
          "passing for a reason unrelated to genuinely sensible motion — e.g. "
          "a different joint configuration reaching a similar angle, or the "
          "EEF taking a path that doesn't actually retrace itself.")
else:
    print("PASS — confirmed independently of our own code: FORGE's own "
          "fingertip tracking shows the gripper genuinely returned to "
          "roughly where it started in real space, not just in joint angle.")

end_pos = last_obs["observation/joint_position"][args.move_joint]
for name, frame_tag in (("exterior_image_1_left", "exterior"), ("wrist_image_left", "wrist")):
    iio.imwrite(f"{args.out}/{frame_tag}_after.png", last_obs[f"observation/{name}"])

forward_drift = mid_pos - start_pos
round_trip_error = end_pos - start_pos
tol = 3 * abs(args.delta)  # generous: a few steps' worth of PD tracking lag

print(f"\nstart_pos={start_pos:+.4f}  mid_pos={mid_pos:+.4f} (forward drift {forward_drift:+.4f})  "
      f"end_pos={end_pos:+.4f} (round-trip error {round_trip_error:+.4f}, tolerance ±{tol:.4f})")

if abs(round_trip_error) <= tol:
    print("PASS — round trip returned near start_pos: this is delta semantics, not absolute targets.")
else:
    print("FAIL — did NOT return near start_pos. If end_pos is instead close to "
          f"0 or to -{args.delta}, _apply_action is very likely applying the "
          "action as an ABSOLUTE target rather than adding it to the current "
          "joint position — check the `self.joint_pos[:, 0:7] + delta_arm` line.")

print(f"Frames saved to {args.out}/ (*_before/_mid/_after.png) — the arm should "
      "visibly move one way then back, compare by eye too.")
print("\nOK if: joint ordering looks sane, the round trip above PASSes, "
      "force/threshold aren't flat zero, and the three image sets visibly differ.")
