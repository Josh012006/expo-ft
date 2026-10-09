"""Scripted ("waypoint") demonstrations for FORGE PegInsert, recorded through the env server.

Runs in .venv (client side, like eval_policy.py), AFTER forge_server.py is up. It drives the SAME
server as the policy does, so every demo is recorded with exactly the observations and the
actions pi05_droid_jointpos sees / produces at evaluation time:

    observation : exterior image, wrist image, 7 joint angles, gripper, (+ force, offset for logs)
    action      : 7 ABSOLUTE joint targets (rad) + gripper command (1 = closed, peg held)

How a demo is generated
-----------------------
The peg is already in the gripper at reset. The only thing to do is bring the peg over the hole
and push it down. We reason about the PEG, not the fingertip, using FORGE's own ground truth:

    offset = (peg base position) - (position of the peg base once fully inserted)      [x, y, z]

(`extra/peg_socket_offset` in the observation). offset = (0, 0, 0) means "inserted". The hole is
`hole_depth` (25 mm) deep, so offset z = 0.025 means "peg base at the top edge of the hole".

Three waypoints, expressed as target offsets, visited in order:

    1. lift   : go straight up until the peg base is `HOVER_MARGIN` above the hole top
                (no sideways motion, so the peg cannot catch on the rim)
    2. align  : at that height, move sideways until x, y offsets are ~0 (peg exactly above the hole)
    3. insert : go straight down, still correcting x, y every step, pressing slightly past the
                final depth so the peg bottoms out

Gripper: closed on the peg for the WHOLE episode, in every demo. Three layers keep it that way:
  - the YAML must say `gripper_can_open: false` (checked at start-up): the server then ignores
    the gripper channel of every action and FORGE keeps its finger target at 0 (closed);
  - the gripper label is the constant 1.0 (closed) in every frame;
  - the finger position is checked at every step: if it moves more than GRIPPER_TOL from its
    value at reset (peg slipped, fingers opened), the attempt is discarded. The check reads the
    finger width in metres through the server's `finger_width` debug op, because the gripper
    OBSERVATION is converted by the server to the model's convention (`gripper_obs` in the YAML)
    and is no longer a width in metres.

Each control step (15 Hz): waypoint -> desired peg translation (capped per step) -> the server
turns it into 7 joint targets with FORGE's own damped-least-squares IK -> `step`.
The label of a frame is the joint-target vector that was actually sent. No relabelling is ever
needed, and the arm tracks it with the same PD controller the policy will use.

Diagnosis tools (added after the first runs, where `align` never finished)
--------------------------------------------------------------------------
  --video-dir DIR     write one mp4 per attempt (exterior | wrist, with step / waypoint / offset
                      printed under the images) for successes AND failures
  --align-tol-mm X    xy error (mm) under which `align` is considered done (default 0.2). Raising it
                      lets the descent start, to see what the peg does at the hole
  --probe             collect nothing: measure how the arm answers to tiny joint commands, to find
                      the dead zone seen at the end of the failed attempts (see run_probe)

Usage (server already running, restarted after the forge_server.py change):
    python scripts/isaaclab/collect_waypoint_demos.py \
        --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --out demos/isaaclab/ForgePegInsert --num-episodes 20
"""
import argparse
import json
import time
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import yaml

from expo_ft.env.env_client import EnvClient

# ---- Waypoint parameters (metres unless stated) -----------------------------------------------
HOVER_MARGIN = 0.010      # peg base is kept this far above the hole's top edge before going in
INSERT_PUSH = 0.002       # aim this far BELOW the final depth: presses the peg to the bottom
ALIGN_TOL_MM = 0.2        # default of --align-tol-mm: "centred over the hole" = xy error below 0.2 mm
MAX_JOINT_STEP = 0.05     # safety: never command a joint more than this (rad) from its measured angle
GRIPPER_CLOSED = 1.0      # 0 = open, 1 = closed (DROID convention); the peg stays held all episode
GRIPPER_TOL = 0.001       # finger position may drift at most 1 mm from its value at reset
HOLD_AFTER_SUCCESS = 5    # keep recording this many steps once inserted, then stop the episode
BANNER_H = 32             # height (px) of the text strip under the video frames (224 + 32 = 256: no resize)


class Waypoint(NamedTuple):
    name: str
    goal: Callable            # offset -> target offset (x, y, z)
    max_xy: float             # largest xy move per control step (m)
    max_z: float              # largest z move per control step (m)
    reached: Callable         # offset -> True when we can move on to the next waypoint


def make_waypoints(hole_depth, align_tol):
    hover_z = hole_depth + HOVER_MARGIN                  # offset z of the "above the hole" point
    return [
        Waypoint("lift",   lambda o: (o[0], o[1], hover_z),  0.0,    0.005,  # keep x, y: straight up
                 lambda o: o[2] >= hover_z - 0.002),
        Waypoint("align",  lambda o: (0.0, 0.0, hover_z),    0.005,  0.003,
                 lambda o: np.linalg.norm(o[:2]) < align_tol and abs(o[2] - hover_z) < 0.002),
        Waypoint("insert", lambda o: (0.0, 0.0, -INSERT_PUSH), 0.0005, 0.0015,  # slow, x, y kept tight
                 lambda o: False),                                              # last one: never "done"
    ]


# Observation entries saved at every step -> name in the .npz file
RECORDED = {
    "observation/exterior_image_1_left": "exterior_image",
    "observation/wrist_image_left": "wrist_image",
    "observation/joint_position": "joint_position",
    "observation/gripper_position": "gripper_position",
    "extra/ft_force": "ft_force",                  # not used by pi05 (kept for the force / tactile work)
    "extra/peg_socket_offset": "peg_socket_offset",
}


def make_debug(client, env_id):
    """Simulation-only helper ops of forge_server.py (a real robot has none of them)."""
    def debug(kind, **kw):
        return client._call_operation("debug", {"env_id": env_id, "kind": kind, **kw})["result"]
    return debug


# ---- Video ------------------------------------------------------------------------------------
def video_frame(obs, text):
    """Exterior and wrist images side by side, with a one-line text strip under them."""
    from PIL import Image, ImageDraw
    ext = np.asarray(obs["observation/exterior_image_1_left"], dtype=np.uint8)
    wrist = np.asarray(obs["observation/wrist_image_left"], dtype=np.uint8)
    images = np.concatenate([ext, wrist], axis=1)
    banner = Image.new("RGB", (images.shape[1], BANNER_H), (0, 0, 0))
    ImageDraw.Draw(banner).text((4, 11), text, fill=(255, 255, 255))
    return np.concatenate([images, np.asarray(banner)], axis=0)


def write_video(path, frames, fps):
    import imageio.v3 as iio      # same call as IsaacLabEnvWrapper uses for the eval videos
    iio.imwrite(str(path), np.stack(frames), fps=fps, codec="libx264")


# ---- One scripted attempt ---------------------------------------------------------------------
def run_episode(client, env_id, seed, geom, args, rng):
    """One scripted attempt. Returns (arrays_to_save, info_dict, video_frames)."""
    debug = make_debug(client, env_id)

    def finger_width():                          # finger joint position in metres (server-side ground truth)
        return float(debug("finger_width"))

    obs, _ = client.reset(env_id, seed)
    width0 = finger_width()                      # grasp width at reset: must not change
    gripper_moved, max_drift = False, 0.0
    quat0 = debug("fingertip_pose")["quat"]      # fingertip orientation to hold for the whole episode
    waypoints = make_waypoints(geom["hole_depth"], args.align_tol_mm * 1e-3)
    wp = 0
    entered = {waypoints[0].name: 0}             # step at which each waypoint was entered (for the log)

    frames = {name: [] for name in RECORDED.values()}
    actions = []
    video = []
    in_a_row = 0                                 # consecutive steps for which FORGE says "inserted"
    done = False
    for t in range(args.max_steps):
        offset = np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64)

        # Move on to the next waypoint as soon as the current one is reached.
        while wp < len(waypoints) - 1 and waypoints[wp].reached(offset):
            wp += 1
            entered[waypoints[wp].name] = t
        w = waypoints[wp]

        # Desired peg translation this step: towards the waypoint, capped per axis.
        limit = np.array([w.max_xy, w.max_xy, w.max_z])
        peg_delta = np.clip(np.asarray(w.goal(offset)) - offset, -limit, limit)
        if args.noise_mm > 0 and w.name != "insert":      # optional variety, never while inserting
            peg_delta[:2] += rng.normal(0.0, args.noise_mm * 1e-3, 2)

        # The peg is rigidly held, so moving the peg by peg_delta = moving the fingertip by peg_delta.
        # The server returns the 7 absolute joint targets that do it (FORGE's own DLS IK).
        q_meas = np.asarray(obs["observation/joint_position"], dtype=np.float64)
        q_ik = np.asarray(debug("ik_joint_target", delta_pos=peg_delta.tolist(), target_quat=quat0))
        q_target = q_meas + np.clip(q_ik - q_meas, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        action = np.concatenate([q_target, [GRIPPER_CLOSED]]).astype(np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError(f"non-finite action at step {t} (seed {seed}): {action}")

        # Record the observation the action was computed from, then act.
        for key, name in RECORDED.items():
            frames[name].append(np.asarray(obs[key]))
        actions.append(action)
        if args.video_dir:
            mm = offset * 1000.0
            video.append(video_frame(
                obs, f"t={t:3d}  {w.name:6s}  xy={np.hypot(mm[0], mm[1]):6.2f} mm  z={mm[2]:5.1f} mm"))
        client.step(env_id, action)
        obs = client.get_observation(env_id)
        done, inserted, _, _ = client.get_info_for_step(env_id)

        # Gripper guard: the fingers must stay where they were at reset (closed on the peg).
        max_drift = max(max_drift, abs(finger_width() - width0))
        gripper_moved = max_drift > GRIPPER_TOL

        in_a_row = in_a_row + 1 if inserted else 0
        if gripper_moved or in_a_row >= HOLD_AFTER_SUCCESS or done:
            break

    arrays = {name: np.stack(vals) for name, vals in frames.items()}
    arrays["actions"] = np.stack(actions)
    # When the episode times out, env.step() has already auto-reset the simulation: `obs` is then the
    # first observation of a NEW episode, not the end of this one. Use the last recorded frame instead.
    final = np.asarray(arrays["peg_socket_offset"][-1] if done else obs["extra/peg_socket_offset"], dtype=np.float64)
    info = {
        "seed": seed,
        "success": in_a_row >= HOLD_AFTER_SUCCESS and not gripper_moved,
        "gripper_moved": gripper_moved,
        "gripper_drift_mm": round(max_drift * 1000, 3),
        "steps": len(actions),
        "final_offset_mm": (final * 1000).round(2).tolist(),
        "max_force_N": float(np.linalg.norm(arrays["ft_force"], axis=1).max()),
        "waypoint_entered_at_step": entered,
    }
    return arrays, info, video


# ---- Probe: how does the arm answer to tiny joint commands? -----------------------------------
PROBE_AMPLITUDES_MRAD = (0.25, 0.5, 1, 2, 5, 10, 20)
PROBE_KICK = 0.020        # rad, "awake" mode: 3 steps of this size just before the measurement


def run_probe(client, env_id, args):
    """Collects nothing. Looks for the dead zone seen at the end of the failed attempts, where a
    command of ~1 mrad per joint produced no motion at all.

    For each amplitude a and each of two modes, the simulation is reset, the arm holds its pose for
    10 steps, then ONE joint (--probe-joint) is commanded at (measured angle + a) for 12 steps. This
    is exactly how the demo script builds its targets (relative to the measured angle at every step).
      rest  : the arm has just been standing still
      awake : same, but 3 steps of +20 mrad are sent right before, so the arm is moving

    Reading the table (reference: a normal answer is ~12% of the command per step, because the PD
    gains of the server give a time constant of about 0.5 s):
      ~12% at every amplitude                    -> no dead zone
      ~0% below some amplitude, ~12% above it    -> dead zone of that size, whatever the history
      'awake' answers but 'rest' does not        -> the arm falls asleep (physics engine sleep threshold)
    """
    j = args.probe_joint

    def advance(obs, delta):
        q = np.asarray(obs["observation/joint_position"], dtype=np.float64)
        target = q.copy()
        target[j] += delta
        client.step(env_id, np.concatenate([target, [GRIPPER_CLOSED]]).astype(np.float32))
        return client.get_observation(env_id)

    def angle(obs):
        return float(np.asarray(obs["observation/joint_position"], dtype=np.float64)[j])

    print(f"\nprobe on joint {j} (0-based); gripper closed; reset seed {args.seed}")
    print("amplitude   mode     answer, steps 1-4   answer, steps 5-12   = % of command")
    print("(mrad)               (mrad per step)     (mrad per step)")
    for amp in PROBE_AMPLITUDES_MRAD:
        for mode in ("rest", "awake"):
            obs, _ = client.reset(env_id, args.seed)
            for _ in range(10):                       # hold the pose
                obs = advance(obs, 0.0)
            if mode == "awake":
                for _ in range(3):
                    obs = advance(obs, PROBE_KICK)
            moves = []
            for _ in range(12):
                before = angle(obs)
                obs = advance(obs, amp * 1e-3)
                moves.append((angle(obs) - before) * 1e3)
            early, late = np.mean(moves[:4]), np.mean(moves[4:])
            print(f"{amp:8.2f}   {mode:5s}   {early:14.3f}   {late:17.3f}   {100 * late / amp:10.0f} %")
    print("\n~12% everywhere: no dead zone | ~0% for small amplitudes only: dead zone | "
          "'awake' answers, 'rest' does not: the arm falls asleep")


# ---- Main -------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Task YAML (host, port, prompt, episode length).")
    p.add_argument("--out", default="demos/isaaclab/ForgePegInsert")
    p.add_argument("--num-episodes", type=int, default=20, help="Successful demos to keep.")
    p.add_argument("--max-attempts", type=int, default=None, help="Default: 2 x num-episodes.")
    p.add_argument("--seed", type=int, default=0, help="Attempt i uses reset seed (seed + i): reproducible.")
    p.add_argument("--max-steps", type=int, default=None, help="Default: max_steps_per_episode of the YAML.")
    p.add_argument("--noise-mm", type=float, default=0.0, help="Gaussian xy jitter (mm) on lift/align moves.")
    p.add_argument("--save-failures", action="store_true", help="Also save failed attempts (as fail_*.npz).")
    p.add_argument("--align-tol-mm", type=float, default=ALIGN_TOL_MM,
                   help="xy error (mm) below which `align` is done and `insert` starts.")
    p.add_argument("--video-dir", default=None, help="If set, write one mp4 per attempt (OK_ / FAIL_) here.")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--probe", action="store_true", help="Run the arm-response probe instead of collecting.")
    p.add_argument("--probe-joint", type=int, default=3, help="Joint (0-6) used by --probe.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("gripper_can_open", True):
        raise SystemExit("gripper_can_open must be false in the task YAML (and the server restarted from it): "
                         "the demos need the gripper closed on the peg for the whole episode.")
    args.max_steps = args.max_steps or cfg.get("max_steps_per_episode", 150)
    max_attempts = args.max_attempts or 2 * args.num_episodes
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)

    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, prompt = client.create_env({})

    if args.probe:
        run_probe(client, env_id, args)
        return

    out.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
    geom = make_debug(client, env_id)("task_geometry")
    print(f"connected: env_id={env_id!r}  hole depth={geom['hole_depth'] * 1000:.0f} mm  "
          f"success: xy<{geom['success_xy'] * 1000:.1f} mm, z<{geom['success_z'] * 1000:.1f} mm  "
          f"align tolerance {args.align_tol_mm:g} mm")

    saved = 0
    attempts = 0
    with open(out / "index.jsonl", "a") as index:
        while saved < args.num_episodes and attempts < max_attempts:
            seed = args.seed + attempts
            attempts += 1
            t0 = time.time()
            arrays, info, video = run_episode(client, env_id, seed, geom, args, rng)
            tag = "OK  " if info["success"] else "FAIL"
            print(f"[{tag}] seed {seed:4d}  {info['steps']:3d} steps  final offset (mm) {info['final_offset_mm']}"
                  f"  max|F| {info['max_force_N']:.1f} N  gripper drift {info['gripper_drift_mm']:.2f} mm"
                  f"  waypoints {info['waypoint_entered_at_step']}"
                  f"  ({time.time() - t0:.0f}s)")
            if info["success"]:
                fname = f"ep_{saved:04d}_seed{seed}.npz"
                saved += 1
            elif args.save_failures:
                fname = f"fail_seed{seed}.npz"
            else:
                fname = None
            if fname:
                np.savez(out / fname, **arrays, prompt=np.array(prompt), seed=np.array(seed))
            if video_dir:
                write_video(video_dir / f"{tag.strip()}_seed{seed}.mp4", video, args.video_fps)
            index.write(json.dumps({**info, "file": fname}) + "\n")
            index.flush()

    print(f"\nkept {saved}/{attempts} attempts as demos  ->  {out}")


if __name__ == "__main__":
    main()
