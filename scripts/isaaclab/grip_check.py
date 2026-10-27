"""Is the peg really clamped? Per reset seed: finger opening, finger motor effort and taxel readings, and how the reset grasp went.

Client, runs in .venv, AFTER forge_server.py is up (robot with taxels: `--set robot=robotiq_2f85_tsf`; `--set robot=franka` works
too, as a control). The arm holds its pose (target = measured angles, gripper closed) for --hold-steps steps, nothing else moves:

    python scripts/isaaclab/grip_check.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml --num-seeds 10
    python scripts/isaaclab/grip_check.py --config ... --num-seeds 100 --hold-steps 3 | grep -E "LIGHT|flagged"

Why. In two demos recorded with the same server, one has the finger motor at its effort limit (16.5 N.m) and a block of
saturated taxels (35 N, 2 columns x 4 rows); the other has 0.0 N.m, 1-10 N and an arch-shaped pattern: the fingers are fully
closed (0.00 mm, 20.39 mm apart) as if the peg were not between the pads. FORGE's reset puts the peg at the fingertip
midpoint with a random offset (PegInsert: held_asset_pos_noise = [3 mm, 0, 3 mm], in the flipped fingertip frame), the
fingers start 1.25 x the peg radius from the axis (about 1 mm of play per side), and the gripper closes for 0.25 s.

Table 1, per seed (width / dist / effort / taxels, right after the reset and after the hold):
    width       finger opening (mm per finger, FORGE's convention); 0.00 = fully closed
    dist        distance between the two finger bodies (mm), from the simulated poses
    effort      PD torque the finger joint is asked for (N.m): ~max = something blocks the finger (the peg), ~0 = the finger
                is at its target and does not squeeze
    taxels      largest zone (N), sum over all zones (N), zones in contact left/right, at the end of the hold
    LIGHT       the seed is flagged when the effort is below --light-effort (N.m)

Table 2 (reset trace, skipped with --no-trace): what the server recorded during the reset's closing, in the frame of the body
`panda_fingertip_centered`, in mm:
    closing axis  unit vector from the right finger to the left finger, in that frame (which of x / y / z the fingers move along)
    placed        where the peg axis is, level with the fingertip midpoint, when FORGE puts it in the hand (before closing):
                  x y z, and `across` = the component along the closing axis (the fingers have ~1 mm of play per side)
    closed        the same after the 0.25 s of closing; `w` = finger opening placed -> closed
For the LIGHT seeds the time series of the closing (every --trace-every physics steps) is printed too.
Reading it: `across` at placement much larger than the play (|across| > ~1 mm) on LIGHT seeds only -> the random placement puts
the peg inside a pad / out of the gap before closing; `across` small and the fingers still close to 0 -> the closing itself
pushes the peg out (the time series shows when).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.env_client import EnvClient  # noqa: E402

CONTACT_N = 0.05          # a zone counts as "in contact" above this force (N)


def closing_axis(row):
    """Unit vector from the right finger to the left finger (fingertip frame), from one trace row."""
    d = np.asarray(row["L"], dtype=np.float64) - np.asarray(row["R"], dtype=np.float64)
    return d / max(np.linalg.norm(d), 1e-9)


def trace_summary(rows):
    """(closing axis, placed row, last row) of a reset trace, or None when it has no placement / step rows."""
    placed = next((r for r in rows if r["ev"] == "place"), None)
    steps = [r for r in rows if r["ev"] == "step"]
    if placed is None or not steps:
        return None
    return closing_axis(placed), placed, steps[-1]


def main():
    p = argparse.ArgumentParser(description="Grip quality per reset seed (finger opening, effort, taxels, reset trace).")
    p.add_argument("--config", required=True, help="Task YAML (host, port).")
    p.add_argument("--seed", type=int, default=0, help="First reset seed.")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--hold-steps", type=int, default=10, help="Steps during which the arm holds its pose.")
    p.add_argument("--light-effort", type=float, default=2.0, help="Effort (N.m) below which a seed is flagged LIGHT.")
    p.add_argument("--no-trace", action="store_true", help="Skip the reset trace (needs the forge_server.py with `trace_next_reset`).")
    p.add_argument("--trace-every", type=int, default=3, help="Time series of the LIGHT seeds: one row every N physics steps.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, _ = client.create_env({})

    def debug(kind):
        res = client._call_operation("debug", {"env_id": env_id, "kind": kind})["result"]
        return None if isinstance(res, dict) and "error" in res else res

    def read(obs):
        tac = obs.get("extra/tactile")
        eff = obs.get("extra/gripper_effort")
        dist = debug("finger_distance")
        width = debug("finger_width")
        out = {"width": None if width is None else float(width) * 1000, "dist": None if dist is None else float(dist) * 1000,
               "effort": None if eff is None else float(np.asarray(eff).reshape(-1)[0])}
        if tac is not None:
            t = np.asarray(tac, dtype=np.float64)
            out.update(tmax=float(t.max()), tsum=float(t.sum()), left=int((t[0] > CONTACT_N).sum()), right=int((t[1] > CONTACT_N).sum()))
        return out

    fmt = lambda v, f: "-" if v is None else format(v, f)
    xyz = lambda v: "(" + ", ".join(format(c, "6.2f") for c in v) + ")"
    print("seed |  width mm (reset -> hold) | dist mm (reset -> hold) | effort N.m (reset -> hold) | taxels max / sum N | zones L/R |")
    flagged, traces, light_rows = [], {}, {}
    for i in range(args.num_seeds):
        seed = args.seed + i
        if not args.no_trace:
            debug("trace_next_reset")
        obs, _ = client.reset(env_id, seed)
        if not args.no_trace:
            res = debug("reset_trace")
            traces[seed] = None if res is None else trace_summary(res["rows"])
            light_rows[seed] = None if res is None else res["rows"]
        a = read(obs)
        for _ in range(args.hold_steps):
            q = np.asarray(obs["observation/joint_position"], dtype=np.float64)
            client.step(env_id, np.concatenate([q, [1.0]]).astype(np.float32))
            obs = client.get_observation(env_id)
        b = read(obs)
        light = b["effort"] is not None and b["effort"] < args.light_effort
        if light:
            flagged.append(seed)
        print(f"{seed:4d} | {fmt(a['width'], '6.2f')} -> {fmt(b['width'], '6.2f')}        | {fmt(a['dist'], '6.2f')} -> {fmt(b['dist'], '6.2f')}"
              f"      | {fmt(a['effort'], '6.2f')} -> {fmt(b['effort'], '6.2f')}          | "
              f"{fmt(b.get('tmax'), '5.1f')} / {fmt(b.get('tsum'), '6.1f')}      | {b.get('left', '-')}/{b.get('right', '-')}       | "
              f"{'LIGHT' if light else ''}")
    print(f"\nflagged LIGHT (effort < {args.light_effort:g} N.m after the hold): {flagged if flagged else 'none'}")

    if args.no_trace:
        return
    if not any(traces.values()):
        print("\nno reset trace (is forge_server.py the version with `trace_next_reset`? restart it after copying the file)")
        return
    print("\nreset trace, frame of panda_fingertip_centered, mm  (across = along the closing axis; the fingers have ~1 mm of play per side)")
    print("seed | closing axis        | placed: peg x, y, z            across | closed: peg x, y, z            across | w placed -> closed")
    for seed, s in traces.items():
        if s is None:
            print(f"{seed:4d} | no trace")
            continue
        c, placed, last = s
        pa, pl = np.asarray(placed["peg"]), np.asarray(last["peg"])
        print(f"{seed:4d} | ({c[0]:5.2f},{c[1]:5.2f},{c[2]:5.2f}) | {xyz(pa)} {float(pa @ c):7.2f} | {xyz(pl)} {float(pl @ c):7.2f} | "
              f"{placed['w_mm']:5.2f} -> {last['w_mm']:5.2f} {'  <- LIGHT' if seed in flagged else ''}")
    for seed in flagged:
        s = traces.get(seed)
        if s is None:
            continue
        c = s[0]
        rows = light_rows[seed]
        print(f"\nseed {seed} (LIGHT): closing, every {args.trace_every} physics steps")
        print("  step |  w mm | effort N.m | peg x, y, z (mm)                 | across | L-R across (mm)")
        for k, r in enumerate(r for r in rows if r["ev"] in ("place", "step")):
            if k % args.trace_every and r["ev"] != "place":
                continue
            peg = np.asarray(r["peg"])
            span = float((np.asarray(r["L"]) - np.asarray(r["R"])) @ c)
            print(f"  {'place' if r['ev'] == 'place' else k:>5} | {r['w_mm']:5.2f} | {fmt(r['effort'], '7.2f'):>10} | {xyz(peg)} | {float(peg @ c):6.2f} | {span:6.2f}")


if __name__ == "__main__":
    main()
