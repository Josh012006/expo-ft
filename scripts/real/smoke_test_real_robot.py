"""Real Franka plumbing smoke test — NO policy, NO model.

Run on the GPU/learner machine (.venv) once the robot server is up on the robot machine:

    python client/run_client.py --config-task-path configs/task/pick_jointpos.py   # robot machine
    python scripts/real/smoke_test_real_robot.py --host localhost --port 8102      # via the SSH tunnel

It is the real-robot counterpart of scripts/isaaclab/smoke_test_env_server.py, and checks, in order:
  0. observation keys / shapes, and the gripper observation convention (0 = open expected)
  1. HOLD: commanding the measured pose repeatedly must not move the arm
  2. ROUND TRIP: one joint ramped by +amplitude then back, as ABSOLUTE joint targets (the same
     interface pi05_droid_jointpos will use), comparing the measured pose to the start pose
  3. (optional, --test-gripper) gripper command direction: 1.0 must CLOSE, 0.0 must OPEN

Every motion is preceded by an explicit confirmation. Keep a hand on the emergency stop.
"""
import argparse
import sys
import time

import numpy as np

from expo_ft.env.env_client import EnvClientWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, default=8102)
parser.add_argument("--hz", type=float, default=15.0)
parser.add_argument("--joint", type=int, default=1, help="Joint index [0-6] for the round trip.")
parser.add_argument("--amplitude", type=float, default=0.08, help="Total excursion of that joint (rad).")
parser.add_argument("--ramp-step", type=float, default=0.01,
                    help="Target increment per control step (rad). The server clamps to its max_joint_step.")
parser.add_argument("--hold-steps", type=int, default=20)
parser.add_argument("--tolerance", type=float, default=0.02, help="Round-trip error accepted as PASS (rad).")
parser.add_argument("--test-gripper", action="store_true")
args = parser.parse_args()


def confirm(message):
    if input(f"\n{message}\nType 'yes' to continue: ").strip().lower() != "yes":
        sys.exit("Aborted by user.")


dt = 1.0 / args.hz
env = EnvClientWrapper(
    {"example_action": np.zeros((1, 8)), "env_usage": "eval", "video_dir": ""},
    host=args.host, port=args.port,
)
print(f"connected — task_description={env.task_description!r}")

confirm("reset() MOVES the robot to its reset pose (and opens the gripper). Clear the workspace.")
obs = env.reset()

# -- 0. observation sanity ------------------------------------------------------------------
print("\nobservation:")
for k, v in obs.items():
    shape = getattr(np.asarray(v), "shape", None)
    print(f"  {k}: shape={shape}" if k != "prompt" else f"  prompt: {v!r}")
for required in ("joint_position", "gripper_position", "exterior_image_1_left", "wrist_image_left"):
    if required not in obs:
        sys.exit(f"MISSING observation key {required!r}: see client/envs/droid_env.py transform_observation.")
q0 = np.asarray(obs["joint_position"], dtype=np.float64).reshape(-1)
g0 = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
assert q0.shape == (7,), f"joint_position should be (7,), got {q0.shape}"
print(f"\nstart joint_position = {np.round(q0, 4)}")
print(f"gripper_position (gripper is OPEN after reset) = {g0:.3f}  "
      "-> expected ~0.0 (model convention: 0 = open). ~1.0 means invert_gripper_observation is NOT active.")


def send(target7, grip):
    """One paced control step with ABSOLUTE joint targets; returns (measured q, executed action)."""
    t0 = time.perf_counter()
    executed, _ = env.step(np.concatenate([target7, [grip]]))
    new_obs = env.get_observation()          # also refreshes the server-side measured state
    left = dt - (time.perf_counter() - t0)
    if left > 0:
        time.sleep(left)
    return np.asarray(new_obs["joint_position"], dtype=np.float64).reshape(-1), np.asarray(executed, dtype=np.float64)


# -- 1. hold --------------------------------------------------------------------------------
confirm(f"[1/3] HOLD: command the current pose for {args.hold_steps} steps (the arm should not move).")
max_drift = 0.0
for _ in range(args.hold_steps):
    q, executed = send(q0, 0.0)          # gripper command 0.0 = stay open
    max_drift = max(max_drift, float(np.abs(q - q0).max()))
print(f"max drift while holding: {max_drift:.4f} rad  ({'PASS' if max_drift < args.tolerance else 'FAIL'})")

# -- 2. single-joint round trip -------------------------------------------------------------
j = args.joint
n = max(1, int(round(args.amplitude / args.ramp_step)))
confirm(f"[2/3] ROUND TRIP: joint[{j}] ramped by {args.amplitude:+.3f} rad in {n} steps, then back.")
target = q0.copy()
peak = 0.0
clamped_steps = 0
for direction in (+1.0, -1.0):
    for _ in range(n):
        target[j] += direction * args.ramp_step
        q, executed = send(target, 0.0)
        peak = max(peak, abs(q[j] - q0[j]))
        clamped_steps += int(np.abs(executed[:7] - target).max() > 1e-6)
for _ in range(15):                          # settle on the start pose
    q, _ = send(q0, 0.0)
err = float(np.abs(q - q0).max())
print(f"peak excursion of joint[{j}]: {peak:.4f} rad (commanded {args.amplitude:.3f})")
print(f"steps where the server's safety clamp altered the target: {clamped_steps}/{2 * n}")
print(f"round-trip error (max over joints): {err:.4f} rad  ({'PASS' if err < args.tolerance else 'FAIL'}, tol {args.tolerance})")

# -- 3. gripper -----------------------------------------------------------------------------
if args.test_gripper:
    confirm("[3/3] GRIPPER: command 1.0 (should CLOSE) for 1 s, then 0.0 (should OPEN). Watch the fingers.")
    for value in (1.0, 0.0):
        for i in range(int(args.hz)):
            q, _ = send(q0, value)
        state = float(np.asarray(env.get_observation()["gripper_position"]).reshape(-1)[0])
        print(f"command {value:.0f} -> observed gripper_position {state:.3f}")
    print("Expected: command 1 -> observation near 1.0 (closed); command 0 -> near 0.0 (open).")

print("\nDone. Compare the PASS/FAIL lines above with what you SAW the arm do.")
