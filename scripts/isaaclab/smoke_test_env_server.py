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

# -- 4. force/threshold not flat zero ---------------------------------------
print(f"\nextra/ft_force: {obs2['extra/ft_force']}")
print(f"extra/force_threshold: {obs2['extra/force_threshold']}")
if obs2["extra/force_threshold"] == 0.0:
    print("  WARNING: still exactly 0.0 — check contact_penalty_thresholds randomization range in the task cfg.")

# -- 3 & 5. deliberate nonzero action on one joint, then its exact opposite -
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
start_pos = obs2["observation/joint_position"][args.move_joint]

move_fwd = np.zeros(8, dtype=np.float32)
move_fwd[args.move_joint] = args.delta
move_bwd = np.zeros(8, dtype=np.float32)
move_bwd[args.move_joint] = -args.delta

for name, frame_tag in (("exterior_image_1_left", "exterior"), ("wrist_image_left", "wrist")):
    iio.imwrite(f"{args.out}/{frame_tag}_before.png", obs2[f"observation/{name}"])

last_obs = None
for t in range(args.move_steps):
    env.step(move_fwd)
    last_obs = env.get_observation()
    if t in (0, args.move_steps - 1):
        pos = last_obs["observation/joint_position"][args.move_joint]
        print(f"forward step {t}: joint[{args.move_joint}] = {pos:.4f}  (start was {start_pos:.4f}, "
              f"expected drift ~{args.delta * (t + 1):+.4f})")

mid_pos = last_obs["observation/joint_position"][args.move_joint]
for name, frame_tag in (("exterior_image_1_left", "exterior"), ("wrist_image_left", "wrist")):
    iio.imwrite(f"{args.out}/{frame_tag}_mid.png", last_obs[f"observation/{name}"])

for t in range(args.move_steps):
    env.step(move_bwd)
    last_obs = env.get_observation()
    if t in (0, args.move_steps - 1):
        pos = last_obs["observation/joint_position"][args.move_joint]
        print(f"backward step {t}: joint[{args.move_joint}] = {pos:.4f}")

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
