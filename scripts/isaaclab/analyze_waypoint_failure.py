"""Diagnose failed waypoint attempts from the .npz files saved by collect_waypoint_demos.py.

Needs only numpy and the saved files (no server, no GPU):

    python scripts/isaaclab/analyze_waypoint_failure.py demos/isaaclab/ForgePegInsert/fail_seed*.npz

For every file it answers three questions, in this order.

1. Where does the peg go?      offset (peg base - fully inserted position, mm) every few steps.
2. In which direction?         at each step, the peg SHOULD move by -offset (xy) / towards the hover
                               height (z). We measure the angle between that wanted direction and the
                               motion that really happened:
                                   ~   0 deg : right direction
                                   ~ 180 deg : sign flipped
                                   ~ +-90 deg or any other constant angle : frame rotated
                                   spread over all angles (low R) : the peg does not follow the command
3. Does the arm follow?        ratio = achieved joint change / commanded joint change over one control step
                               (1/15 s). ~1 : the PD controller tracks; << 1 : it lags far behind.

4. (end of episode)            commanded vs achieved joint change over the last steps, to tell a frozen peg
                               caused by a zero command from one caused by an arm that does not move.

The files hold the observation BEFORE each action, so step t's effect is offset[t+1] - offset[t].
The z statistic assumes the peg is still aiming at the hover height: it is only meaningful for
attempts that never reached the "insert" waypoint (all the failures seen so far).
"""
import argparse

import numpy as np

HOVER_MM = 35.0        # hole depth (25) + HOVER_MARGIN (10) of collect_waypoint_demos.py
MAX_JOINT_STEP = 0.05  # same safety clip as the collection script (rad)


def circular_mean_deg(angles_rad):
    """Mean direction (deg) and concentration R in [0, 1] (1 = all identical, 0 = uniformly spread)."""
    s, c = np.sin(angles_rad).mean(), np.cos(angles_rad).mean()
    return np.degrees(np.arctan2(s, c)), float(np.hypot(s, c))


def analyze(path, every, tail):
    d = np.load(path)
    off = d["peg_socket_offset"].astype(np.float64) * 1000.0   # (T, 3) mm
    q = d["joint_position"].astype(np.float64)                 # (T, 7) rad
    act = d["actions"].astype(np.float64)                      # (T, 8)
    T = len(off)
    print(f"\n=== {path}   ({T} steps) ===")

    # 1. trajectory ---------------------------------------------------------------------------
    print("1) offset (mm)        step      x        y        z     |xy|")
    for t in sorted(set(list(range(0, T, every)) + [T - 1])):
        x, y, z = off[t]
        print(f"                    {t:5d} {x:8.2f} {y:8.2f} {z:8.2f} {np.hypot(x, y):8.2f}")

    # 2. direction ----------------------------------------------------------------------------
    delta = off[1:] - off[:-1]                                 # what each step really did
    xy, dxy = off[:-1, :2], delta[:, :2]
    far = np.hypot(*xy.T) > 3.0                                # direction to the hole is well defined
    moved = np.hypot(*dxy.T) > 1e-3
    m = far & moved
    print("2) direction")
    if m.sum() >= 5:
        want = -xy[m]                                          # to reduce the xy offset
        got = dxy[m]
        ang = np.arctan2(want[:, 0] * got[:, 1] - want[:, 1] * got[:, 0], (want * got).sum(1))
        mean_deg, R = circular_mean_deg(ang)
        print(f"   xy: angle wanted->real = {mean_deg:+7.1f} deg   concentration R = {R:.2f}   "
              f"median |step| = {np.median(np.hypot(*got.T)):.3f} mm/step   ({m.sum()} steps)")
    else:
        print(f"   xy: not enough steps with |xy| > 3 mm and visible motion ({m.sum()})")
    err_z = HOVER_MM - off[:-1, 2]
    mz = np.abs(err_z) > 2.0
    if mz.sum() >= 5:
        agree = float((np.sign(delta[mz, 2]) == np.sign(err_z[mz])).mean())
        print(f"   z : moves towards the hover height ({HOVER_MM:.0f} mm) in {100 * agree:.0f}% of steps   "
              f"median |step| = {np.median(np.abs(delta[mz, 2])):.3f} mm/step   ({mz.sum()} steps)")
    else:
        print(f"   z : within 2 mm of the hover height almost all the time ({mz.sum()} steps outside)")

    # 3. tracking -----------------------------------------------------------------------------
    cmd = act[:-1, :7] - q[:-1]                                # joint change we asked for
    ach = q[1:] - q[:-1]                                       # joint change we got
    ratio = (cmd * ach).sum(0) / np.maximum((cmd * cmd).sum(0), 1e-12)
    clipped = float((np.abs(cmd).max(1) >= MAX_JOINT_STEP - 1e-6).mean())
    print("3) arm tracking (achieved / commanded joint change per control step)")
    print("   per joint: " + "  ".join(f"{r:5.2f}" for r in ratio))
    print(f"   largest commanded joint change = {np.abs(cmd).max():.3f} rad "
          f"(clip {MAX_JOINT_STEP} rad hit in {100 * clipped:.0f}% of steps)")

    # 4. the end of the episode ---------------------------------------------------------------
    n = min(tail, len(cmd))
    drift = off[-1] - off[-1 - n]
    print(f"4) last {n} steps (is the peg frozen, and why?)")
    print(f"   offset change over these steps = ({drift[0]:+.3f}, {drift[1]:+.3f}, {drift[2]:+.3f}) mm")
    print("   mean |commanded joint change| (mrad): " + "  ".join(f"{1e3 * v:5.2f}" for v in np.abs(cmd[-n:]).mean(0)))
    print("   mean |achieved  joint change| (mrad): " + "  ".join(f"{1e3 * v:5.2f}" for v in np.abs(ach[-n:]).mean(0)))
    print("   -> commanded ~ 0       : the IK sees no error although the offset is not 0 (bias in the target)")
    print("   -> commanded > 0, achieved ~ 0 : the arm is stuck (friction / dead zone of the PD loop)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="fail_seed*.npz or ep_*.npz")
    p.add_argument("--every", type=int, default=15, help="print the offset every N steps")
    p.add_argument("--tail", type=int, default=50, help="length of the end-of-episode window of section 4")
    args = p.parse_args()
    for f in args.files:
        analyze(f, args.every, args.tail)


if __name__ == "__main__":
    main()
