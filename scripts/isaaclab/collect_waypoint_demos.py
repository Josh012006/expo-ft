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

Four waypoints, expressed as target offsets, visited in order:

    1. lift    : go straight up until the peg base is `HOVER_MARGIN` above the hole top
                 (no sideways motion, so the peg cannot catch on the rim)
    2. align   : at that height, move sideways until x, y offsets are ~0 (peg exactly above the hole)
    3. descend : go straight down FAST to DESCEND_STOP above the hole's top edge, x, y kept tight; it hands
                 over to `insert` only once the xy error is below DESCEND_HANDOFF_XY (otherwise it waits
                 there and corrects), so the peg never reaches the rim off-centre
    4. insert  : go down SLOWLY inside the hole, still correcting x, y every step, pressing slightly past the
                 final depth so the peg bottoms out

Speed of the phases. Each waypoint caps the peg move per control step. The arm only follows a fraction of
each commanded change (~36% with the 400 / 60 PD gains of the YAML), so the real speed is about cap x 0.36 and,
more importantly for SFT, the joint-target offsets the policy must learn (target - measured angle) scale with
the cap. The first version had lift z 5 mm, align xy 5 / z 3 mm, no `descend` phase (`insert`, 1.5 mm in z,
covered the whole descent): ~70% of the frames were slow, tiny offsets. Defaults are now faster (see the
--*-max-* flags). Reproduce the first version approximately with:
    --lift-max-z-mm 5 --align-max-xy-mm 5 --align-max-z-mm 3 --descend-max-z-mm 1.5 --descend-max-xy-mm 0.5

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
  --lift-max-z-mm, --align-max-xy-mm, --align-max-z-mm, --descend-max-xy-mm, --descend-max-z-mm,
  --insert-max-xy-mm, --insert-max-z-mm
                      per-step caps (mm) of each phase, see "Speed of the phases"; the number of steps where a
                      joint target had to be clipped to MAX_JOINT_STEP is printed / stored as `clipped_steps`
                      (if it is not ~0 the caps are too high: the direction of the move gets distorted)
  --noise-mm, --hover-jitter-mm, --speed-jitter
                      variety between demos (all 0 = every demo from the same start follows the same path);
                      the values drawn are written to index.jsonl. The start states already differ from
                      one seed to the next. How many demos: --num-episodes (kept) / --max-attempts.
                      (--noise-mm jitters the desired move and the label contains it: that is NOT the noise
                      injection below, whose labels stay clean.)
  --align-tol-mm X    xy error (mm) under which `align` is considered done (default 0.2). Raising it
                      lets the descent start, to see what the peg does at the hole
  --probe             collect nothing: measure how the arm answers to tiny joint commands, to find
                      the dead zone seen at the end of the failed attempts (see run_probe)

Noise injection (arXiv 2507.09061, "Practice 2" / Algorithm 2) -- off by default
---------------------------------------------------------------------------------
A share --noise-fraction of the kept demos is recorded with noise on the arm: the joint targets SENT to the
simulation are  label + sigma * z  (z uniform in the 7-d unit ball), but the saved `actions` stay the CLEAN label,
i.e. what the expert would command from the state it actually reached. The policy therefore sees perturbed
states and learns the expert's reaction to them (recovery), and the noise never enters a label. The gripper is
never noised. --noise-fraction 0 (default) = exactly the previous behaviour, same files, same index.jsonl.
(In the paper alpha is the share of CLEAN trajectories; here it is the share of NOISY ones.) Which demos are
noisy is decided per demo, not per step; clean and noisy demos are interleaved, so the first N files of a
folder keep the mix (the SFT conversion takes the first --max-episodes files).

  --noise-mode burst  (default) --noise-bursts bursts of --noise-burst-len steps, first one starting at a random
                      step in [1, --noise-start-max]; the rest of the demo is clean, so the frames after a
                      burst show the expert recovering with an exact label
  --noise-mode white  noise at every step (what the paper does for a policy that predicts ONE action). With
                      chunks the label of a frame (the next 16 actions) is hard to keep consistent: prefer small sigma
  --action-noise-sigma  radius of the noise (rad per step, all 7 joints). Default 0.01 (burst) / 0.003 (white).
                      The arm follows ~36% of a command per step, so this moves the peg by a few mm: calibrate it
                      with a pilot (peak_xy_lift_align_mm and the success rate in the summary / index.jsonl)
  --noise-insert-scale  noise multiplier in `descend` / `insert` (default 0: noise only in `lift` / `align`, never
                      near or in the hole); 1 = noise during the whole demo
  --noise-off-xy-mm   no noise in `align` once the xy error is below this (default 0 = off). Useful with white
                      noise: the 0.2 mm align tolerance otherwise fights the noise and `align` lasts much longer

A noisy demo is saved as ep_NNNN_seedS_noisy.npz with two extra arrays, `noise` (T, 7) = what was added to the
label (after the MAX_JOINT_STEP clip) and `noisy`; clean demos get a zero `noise` array. The conversion and
norm-stats scripts read only the usual arrays, so they work on these folders unchanged. replay_demos.py replays
the clean labels, so it does not reproduce the noise. Use a separate --out folder for a noisy dataset.

Usage (server already running, restarted after the forge_server.py change):
    python scripts/isaaclab/collect_waypoint_demos.py \
        --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --out demos/isaaclab/ForgePegInsert --num-episodes 20
    # with noise: 35% of the 50 demos get a 4-step burst
    python scripts/isaaclab/collect_waypoint_demos.py \
        --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --out demos/isaaclab/ForgePegInsertNoise --num-episodes 50 --noise-fraction 0.35
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
DESCEND_STOP = 0.002      # the fast descent ends this far above the hole's top edge
DESCEND_HANDOFF_XY = 0.001  # ... and only if the peg is within 1 mm (xy) of the hole axis
MAX_JOINT_STEP = 0.05     # safety: never command a joint more than this (rad) from its measured angle
GRIPPER_CLOSED = 1.0      # 0 = open, 1 = closed (DROID convention); the peg stays held all episode
GRIPPER_TOL = 0.001       # finger position may drift at most 1 mm from its value at reset
HOLD_AFTER_SUCCESS = 5    # keep recording this many steps once inserted, then stop the episode
MIN_HOVER_MARGIN = 0.004  # smallest hover height above the hole's top edge, whatever the jitter
BANNER_H = 32             # height (px) of the text strip under the video frames (448 + 32 = 480: a multiple of 16, no resize)
NOISE_STREAM = 0x4E015E   # second entry of the per-attempt noise seed [seed, NOISE_STREAM]: own stream, never touches `rng`
NOISE_PHASES = ("lift", "align")   # the only phases with full noise (the others get --noise-insert-scale)
SIGMA_DEFAULT = {"burst": 0.01, "white": 0.003}   # rad, see --action-noise-sigma


class Waypoint(NamedTuple):
    name: str
    goal: Callable            # offset -> target offset (x, y, z)
    max_xy: float             # largest xy move per control step (m)
    max_z: float              # largest z move per control step (m)
    reached: Callable         # offset -> True when we can move on to the next waypoint


class Caps(NamedTuple):
    """Largest peg move per control step (m) of each phase. The defaults are the fast version."""
    lift_z: float = 0.015
    align_xy: float = 0.010
    align_z: float = 0.008
    descend_xy: float = 0.001
    descend_z: float = 0.008
    insert_xy: float = 0.0005
    # 3 mm, above INSERT_PUSH (2 mm): the cap no longer clips the goal, so the step is the full 2 mm press. With 1.5 mm
    # the push (max|F| ~2.6-3.3 N) was too weak for a peg held tilted: seed 1 (3.8 deg) stopped 5-7 mm short of the
    # bottom in 4 of 4 runs and seed 111 (3.4 deg) 1.7 mm short in 1 of 1; with 3 mm they went in in 4 of 4 and 1 of 1
    # (max|F| ~5-6 N). Seeds up to 2.9 deg went in with either value.
    insert_z: float = 0.003


def make_waypoints(hole_depth, align_tol, hover_margin=HOVER_MARGIN, speed=1.0, caps=Caps()):
    """speed scales the per-step caps of `lift`, `align` and `descend`; `insert` keeps its slow caps."""
    hover_z = hole_depth + hover_margin                  # offset z of the "above the hole" point
    stop_z = hole_depth + DESCEND_STOP                   # where the fast descent ends
    return [
        Waypoint("lift",    lambda o: (o[0], o[1], hover_z),  0.0,  caps.lift_z * speed,  # keep x, y: straight up
                 lambda o: o[2] >= hover_z - 0.002),
        Waypoint("align",   lambda o: (0.0, 0.0, hover_z),    caps.align_xy * speed,  caps.align_z * speed,
                 lambda o: np.linalg.norm(o[:2]) < align_tol and abs(o[2] - hover_z) < 0.002),
        Waypoint("descend", lambda o: (0.0, 0.0, stop_z),     caps.descend_xy * speed,  caps.descend_z * speed,
                 lambda o: o[2] <= stop_z + 0.002 and np.linalg.norm(o[:2]) < DESCEND_HANDOFF_XY),
        Waypoint("insert",  lambda o: (0.0, 0.0, -INSERT_PUSH), caps.insert_xy, caps.insert_z,  # slow, x, y tight
                 lambda o: False),                                                # last one: never "done"
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
# Saved only when the server sends it (robot profile with taxels, `robot: robotiq_2f85_tsf`): (T, 2, 7, 4) N, [:, 0] left
# finger, [:, 1] right finger, row 0 = fingertip. Demos of other robots simply have no "tactile" array.
RECORDED_OPTIONAL = {
    "extra/tactile": "tactile",
    "extra/gripper_effort": "gripper_effort",      # Robotiq only: PD torque demanded of the finger joint (N.m), scalar per step
}


def make_debug(client, env_id):
    """Simulation-only helper ops of forge_server.py (a real robot has none of them)."""
    def debug(kind, **kw):
        return client._call_operation("debug", {"env_id": env_id, "kind": kind, **kw})["result"]
    return debug


# ---- Noise injection --------------------------------------------------------------------------
def unit_ball(rng, dim):
    """One point drawn uniformly in the unit ball of R^dim (the z of Algorithm 2)."""
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v) * rng.uniform() ** (1.0 / dim)


def burst_steps(rng, n_bursts, length, gap, start_max):
    """Steps of the noise bursts: the first starts at a random step in [1, start_max], each next one `gap` (+0..3)
    steps after the end of the previous one. A burst that falls in `descend` / `insert` is silent (scale 0)."""
    steps, s = set(), int(rng.integers(1, start_max + 1))
    for _ in range(n_bursts):
        steps.update(range(s, s + length))
        s += length + gap + int(rng.integers(0, 4))
    return steps


# ---- Video ------------------------------------------------------------------------------------
def video_frame(obs, text, colour=(255, 255, 255)):
    """Exterior | wrist side by side, the taxel panel below them (448 x 448), a one-line text strip under it all."""
    from expo_ft.env.video_panels import compose_frame, panel_kwargs, text_banner
    frame = compose_frame(obs["observation/exterior_image_1_left"], obs["observation/wrist_image_left"], **panel_kwargs(obs))
    banner = text_banner([(text, colour)], frame.shape[1], BANNER_H)
    return frame if banner is None else np.concatenate([frame, banner], axis=0)


def write_video(path, frames, fps):
    import imageio.v3 as iio      # same call as IsaacLabEnvWrapper uses for the eval videos
    iio.imwrite(str(path), np.stack(frames), fps=fps, codec="libx264")


# ---- One scripted attempt ---------------------------------------------------------------------
def run_episode(client, env_id, seed, geom, args, rng, noisy=False):
    """One scripted attempt. Returns (arrays_to_save, info_dict, video_frames).
    noisy=True: noise is added to the joint targets SENT to the arm (see "Noise injection"); the recorded action
    stays the clean label."""
    debug = make_debug(client, env_id)
    noise_rng = np.random.default_rng([seed, NOISE_STREAM])         # separate stream: `rng` is used as before
    bursts = (burst_steps(noise_rng, args.noise_bursts, args.noise_burst_len, args.noise_burst_gap, args.noise_start_max)
              if noisy and args.noise_mode == "burst" else set())
    noise_log, noise_steps, peak_xy = [], 0, 0.0

    def finger_width():                          # finger joint position in metres (server-side ground truth)
        return float(debug("finger_width"))

    obs, _ = client.reset(env_id, seed)
    width0 = finger_width()                      # grasp width at reset: must not change
    gripper_moved, max_drift = False, 0.0
    quat0 = debug("fingertip_pose")["quat"]      # fingertip orientation to hold for the whole episode
    # Per-attempt variety (all off by default): a different hover height and a different speed of the
    # approach. The hover margin never goes below MIN_HOVER_MARGIN so the peg cannot touch the rim.
    hover_margin = max(MIN_HOVER_MARGIN, (args.hover_margin_mm + rng.uniform(-1, 1) * args.hover_jitter_mm) * 1e-3)
    speed = 1.0 + rng.uniform(-1, 1) * args.speed_jitter
    caps = Caps(
        lift_z=args.lift_max_z_mm * 1e-3, align_xy=args.align_max_xy_mm * 1e-3, align_z=args.align_max_z_mm * 1e-3,
        descend_xy=args.descend_max_xy_mm * 1e-3, descend_z=args.descend_max_z_mm * 1e-3,
        insert_xy=args.insert_max_xy_mm * 1e-3, insert_z=args.insert_max_z_mm * 1e-3)
    waypoints = make_waypoints(geom["hole_depth"], args.align_tol_mm * 1e-3, hover_margin, speed, caps)
    wp = 0
    entered = {waypoints[0].name: 0}             # step at which each waypoint was entered (for the log)

    recorded = {**RECORDED, **{k: v for k, v in RECORDED_OPTIONAL.items() if k in obs}}
    frames = {name: [] for name in recorded.values()}
    actions = []
    video = []
    in_a_row = 0                                 # consecutive steps for which FORGE says "inserted"
    clipped_steps = 0                            # steps where a joint target hit MAX_JOINT_STEP
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
        if args.noise_mm > 0 and w.name in ("lift", "align"):   # optional variety, never near / in the hole
            peg_delta[:2] += rng.normal(0.0, args.noise_mm * 1e-3, 2)

        # The peg is rigidly held, so moving the peg by peg_delta = moving the fingertip by peg_delta.
        # The server returns the 7 absolute joint targets that do it (FORGE's own DLS IK).
        q_meas = np.asarray(obs["observation/joint_position"], dtype=np.float64)
        q_ik = np.asarray(debug("ik_joint_target", delta_pos=peg_delta.tolist(), target_quat=quat0))
        # --cmd-gain > 1 compensates a sluggish arm: it follows only a fraction of each (relative) target
        # change, so asking for G times more makes the peg move at about the intended speed.
        dq = args.cmd_gain * (q_ik - q_meas)
        clipped_steps += int(np.abs(dq).max() > MAX_JOINT_STEP)
        q_target = q_meas + np.clip(dq, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        action = np.concatenate([q_target, [GRIPPER_CLOSED]]).astype(np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError(f"non-finite action at step {t} (seed {seed}): {action}")

        # Noise injection: what is SENT is the clean target + noise (kept within MAX_JOINT_STEP of the measured
        # angles); what is RECORDED stays `action`. Noise only in lift / align unless --noise-insert-scale > 0.
        sent = action
        noise = np.zeros(7)
        if noisy:
            scale = 1.0 if w.name in NOISE_PHASES else args.noise_insert_scale
            xy_mm = float(np.hypot(offset[0], offset[1])) * 1000.0
            quiet = w.name == "align" and args.noise_off_xy_mm > 0 and xy_mm < args.noise_off_xy_mm
            if scale > 0 and not quiet and (args.noise_mode == "white" or t in bursts):
                noisy_target = q_target + args.action_noise_sigma * scale * unit_ball(noise_rng, 7)
                noisy_target = np.clip(noisy_target, q_meas - MAX_JOINT_STEP, q_meas + MAX_JOINT_STEP)
                noise = noisy_target - q_target
                sent = np.concatenate([noisy_target, [GRIPPER_CLOSED]]).astype(np.float32)
            noise_steps += int(np.any(noise != 0.0))
            noise_log.append(noise)
        if w.name in NOISE_PHASES:
            peak_xy = max(peak_xy, float(np.hypot(offset[0], offset[1])))

        # Record the observation the action was computed from, then act.
        for key, name in recorded.items():
            frames[name].append(np.asarray(obs[key]))
        actions.append(action)
        if args.video_dir:
            mm = offset * 1000.0
            text = f"t={t:3d}  {w.name:6s}  xy={np.hypot(mm[0], mm[1]):6.2f} mm  z={mm[2]:5.1f} mm"
            if np.any(noise != 0.0):
                video.append(video_frame(obs, f"{text}  NOISE {np.linalg.norm(noise) * 1e3:4.1f} mrad", (255, 170, 40)))
            else:
                video.append(video_frame(obs, text + ("  [noisy demo]" if noisy else "")))
        client.step(env_id, sent)
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
    if args.noise_fraction > 0:                  # only when the option is used: fraction 0 saves exactly the old files
        arrays["noise"] = np.stack(noise_log).astype(np.float32) if noise_log else np.zeros((len(actions), 7), np.float32)
        arrays["noisy"] = np.array(bool(noisy))
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
        **({"max_tactile_N": float(arrays["tactile"].max())} if "tactile" in arrays else {}),
        "waypoint_entered_at_step": entered,
        "clipped_steps": clipped_steps,
        "hover_margin_mm": round(hover_margin * 1000, 2),
        "approach_speed_factor": round(speed, 3),
    }
    if args.noise_fraction > 0:
        info.update({
            "noisy": bool(noisy),
            "noise_mode": args.noise_mode if noisy else None,
            "noise_sigma_rad": args.action_noise_sigma if noisy else 0.0,
            "noise_steps": noise_steps,                                   # steps where the arm really got noise
            "noise_burst_steps": sorted(bursts),
            "peak_xy_lift_align_mm": round(peak_xy * 1000, 2),            # largest xy error while in lift / align
        })
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

    Reading the table (the normal answer is a fixed share of the command per step, set by the PD gains of
    the server: ~12% with stiffness 80 / damping 40, because the time constant is about 0.5 s):
      the same % at every amplitude              -> no dead zone
      ~0% below some amplitude, normal above it  -> dead zone of that size, whatever the history
      'awake' answers but 'rest' does not        -> the arm falls asleep (tested: it does not)
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
    print("\nsame % at every amplitude: no dead zone | ~0% for small amplitudes only: dead zone")


# ---- Main -------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Task YAML (host, port, prompt, episode length).")
    p.add_argument("--out", default="demos/isaaclab/ForgePegInsert")
    p.add_argument("--num-episodes", type=int, default=20, help="Successful demos to keep.")
    p.add_argument("--max-attempts", type=int, default=None, help="Default: 2 x num-episodes.")
    p.add_argument("--seed", type=int, default=0, help="Attempt i uses reset seed (seed + i): reproducible.")
    p.add_argument("--max-steps", type=int, default=None, help="Default: max_steps_per_episode of the YAML.")
    p.add_argument("--noise-mm", type=float, default=0.0,
                   help="Gaussian xy jitter (mm) on lift/align moves; the label contains it (variety, not noise injection).")
    p.add_argument("--noise-fraction", type=float, default=0.0,
                   help="Share of the kept demos recorded WITH noise injection (0 = none, as before; e.g. 0.35). "
                        "Their labels stay clean. See 'Noise injection' at the top of the file.")
    p.add_argument("--noise-mode", choices=("burst", "white"), default="burst",
                   help="burst: a few short bursts then clean (default); white: noise at every step.")
    p.add_argument("--action-noise-sigma", type=float, default=None,
                   help="Noise radius (rad per step, 7 joints, uniform in the unit ball). Default 0.01 (burst) / 0.003 (white).")
    p.add_argument("--noise-burst-len", type=int, default=4, help="Steps per burst (burst mode).")
    p.add_argument("--noise-bursts", type=int, default=1,
                   help="Bursts per noisy demo (burst mode). Noise only acts in lift/align, ~12 steps: more than 1 rarely fits.")
    p.add_argument("--noise-burst-gap", type=int, default=8, help="Clean steps between two bursts (+0..3).")
    p.add_argument("--noise-start-max", type=int, default=6, help="The first burst starts at a random step in [1, this].")
    p.add_argument("--noise-insert-scale", type=float, default=0.0,
                   help="Noise multiplier in descend/insert (0 = none there, 1 = same as lift/align).")
    p.add_argument("--noise-off-xy-mm", type=float, default=0.0,
                   help="No noise in `align` once the xy error is below this many mm (0 = disabled).")
    p.add_argument("--hover-margin-mm", type=float, default=HOVER_MARGIN * 1000,
                   help="Height of the peg base above the hole's top edge before the descent (mm).")
    p.add_argument("--hover-jitter-mm", type=float, default=0.0,
                   help="Each attempt draws its hover height uniformly in [margin - j, margin + j] (min 4 mm).")
    p.add_argument("--speed-jitter", type=float, default=0.0,
                   help="Each attempt scales the lift/align/descend speed by a factor drawn in [1 - j, 1 + j] (e.g. 0.3).")
    c = Caps()
    p.add_argument("--lift-max-z-mm", type=float, default=c.lift_z * 1e3, help="Largest z move per step in `lift` (mm).")
    p.add_argument("--align-max-xy-mm", type=float, default=c.align_xy * 1e3, help="Largest xy move per step in `align` (mm).")
    p.add_argument("--align-max-z-mm", type=float, default=c.align_z * 1e3, help="Largest z move per step in `align` (mm).")
    p.add_argument("--descend-max-xy-mm", type=float, default=c.descend_xy * 1e3,
                   help="Largest xy move per step in the fast `descend` (mm).")
    p.add_argument("--descend-max-z-mm", type=float, default=c.descend_z * 1e3,
                   help="Largest z move per step in the fast `descend` (mm).")
    p.add_argument("--insert-max-xy-mm", type=float, default=c.insert_xy * 1e3,
                   help="Largest xy move per step in `insert`, inside the hole (mm).")
    p.add_argument("--insert-max-z-mm", type=float, default=c.insert_z * 1e3,
                   help="Largest z move per step in `insert`, inside the hole (mm). Raise it (e.g. 3) to shorten "
                        "the slow part, if the peg does not jam.")
    p.add_argument("--save-failures", action="store_true", help="Also save failed attempts (as fail_*.npz).")
    p.add_argument("--align-tol-mm", type=float, default=ALIGN_TOL_MM,
                   help="xy error (mm) below which `align` is done and `insert` starts.")
    p.add_argument("--cmd-gain", type=float, default=1.0,
                   help="Multiplies every joint-target change sent to the arm (default 1 = unchanged).")
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
    if not 0.0 <= args.noise_fraction <= 1.0:
        raise SystemExit(f"--noise-fraction must be in [0, 1], got {args.noise_fraction}")
    if args.action_noise_sigma is None:
        args.action_noise_sigma = SIGMA_DEFAULT[args.noise_mode]
    if args.noise_fraction > 0 and (args.action_noise_sigma <= 0 or args.noise_burst_len < 1 or args.noise_bursts < 1
                                    or args.noise_start_max < 1 or args.noise_insert_scale < 0):
        raise SystemExit("noise injection needs --action-noise-sigma > 0, --noise-burst-len >= 1, --noise-bursts >= 1, "
                         "--noise-start-max >= 1 and --noise-insert-scale >= 0")
    max_attempts = args.max_attempts or 2 * args.num_episodes
    n_noisy = int(round(args.noise_fraction * args.num_episodes))
    quota = {"clean": args.num_episodes - n_noisy, "noisy": n_noisy}      # successful demos wanted of each kind
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
          f"align tolerance {args.align_tol_mm:g} mm  command gain {args.cmd_gain:g}")
    print(f"per-step caps (mm): lift z {args.lift_max_z_mm:g} | align xy {args.align_max_xy_mm:g} z {args.align_max_z_mm:g}"
          f" | descend xy {args.descend_max_xy_mm:g} z {args.descend_max_z_mm:g}"
          f" | insert xy {args.insert_max_xy_mm:g} z {args.insert_max_z_mm:g}   hover margin {args.hover_margin_mm:g} mm")

    if args.noise_fraction > 0:
        what = (f"{args.noise_bursts} burst(s) of {args.noise_burst_len} steps" if args.noise_mode == "burst"
                else "noise at every step")
        print(f"noise injection: {quota['noisy']} noisy + {quota['clean']} clean demos wanted | mode {args.noise_mode} ({what}) "
              f"| sigma {args.action_noise_sigma:g} rad | scale in descend/insert {args.noise_insert_scale:g} "
              f"| labels stay clean")

    saved = 0
    attempts = 0
    step_counts = []                                  # length of the successful attempts
    kinds = {k: {"tried": 0, "ok": 0, "steps": [], "peak": [], "align": []} for k in quota}
    with open(out / "index.jsonl", "a") as index:
        while saved < args.num_episodes and attempts < max_attempts:
            seed = args.seed + attempts
            attempts += 1
            # clean / noisy interleaved: the kind that is furthest behind its quota goes next (all clean at fraction 0)
            kind = min((k for k in quota if kinds[k]["ok"] < quota[k]), key=lambda k: kinds[k]["ok"] / quota[k])
            noisy = kind == "noisy"
            kinds[kind]["tried"] += 1
            t0 = time.time()
            arrays, info, video = run_episode(client, env_id, seed, geom, args, rng, noisy=noisy)
            tag = "OK  " if info["success"] else "FAIL"
            extra = f"  [noisy: {info['noise_steps']} noise steps]" if noisy else ""
            print(f"[{tag}] seed {seed:4d}  {info['steps']:3d} steps  final offset (mm) {info['final_offset_mm']}"
                  f"  max|F| {info['max_force_N']:.1f} N  gripper drift {info['gripper_drift_mm']:.2f} mm"
                  f"  waypoints {info['waypoint_entered_at_step']}  clipped {info['clipped_steps']}{extra}"
                  f"  ({time.time() - t0:.0f}s)")
            if info["success"]:
                fname = f"ep_{saved:04d}_seed{seed}{'_noisy' if noisy else ''}.npz"
                saved += 1
                step_counts.append(info["steps"])
                kinds[kind]["ok"] += 1
                kinds[kind]["steps"].append(info["steps"])
                if "peak_xy_lift_align_mm" in info:
                    kinds[kind]["peak"].append(info["peak_xy_lift_align_mm"])
                    entered = info["waypoint_entered_at_step"]
                    kinds[kind]["align"].append(entered.get("descend", info["steps"]))   # steps spent before `descend`
            elif args.save_failures:
                fname = f"fail_seed{seed}{'_noisy' if noisy else ''}.npz"
            else:
                fname = None
            if fname:
                np.savez(out / fname, **arrays, prompt=np.array(prompt), seed=np.array(seed))
            if video_dir:
                write_video(video_dir / f"{tag.strip()}_seed{seed}{'_noisy' if noisy else ''}.mp4", video, args.video_fps)
            index.write(json.dumps({**info, "file": fname}) + "\n")
            index.flush()

    print(f"\nkept {saved}/{attempts} attempts as demos  ->  {out}")
    if step_counts:
        print(f"successful attempts: {len(step_counts)}, mean length {np.mean(step_counts):.0f} steps "
              f"(max allowed {args.max_steps})")
    if args.noise_fraction > 0:
        print("\nclean vs noisy (success rate = kept / attempted of that kind; 'before descend' = steps spent in lift + align)")
        for k, s in kinds.items():
            if s["tried"] == 0:
                continue
            print(f"  {k:5s}: {s['ok']}/{s['tried']} ok ({100 * s['ok'] / s['tried']:.0f}%)"
                  f" | mean length {np.mean(s['steps']):.0f} steps | before descend {np.mean(s['align']):.1f} steps"
                  f" | mean peak xy in lift/align {np.mean(s['peak']):.2f} mm" if s["ok"] else
                  f"  {k:5s}: 0/{s['tried']} ok")


if __name__ == "__main__":
    main()
