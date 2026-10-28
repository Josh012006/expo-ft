"""How far does the injected noise really move the peg?  (clean demo vs noisy demo of the SAME seed)

No server, no GPU: numpy only, runs in .venv.

    python scripts/isaaclab/noise_effect.py \
        --noisy demos/isaaclab/ForgePegInsert_Robotiq_noise_pilot \
        --clean demos/isaaclab/ForgePegInsert_Robotiq_scripted

Why: `peak_xy_lift_align_mm` of index.jsonl is the largest xy error seen in lift / align, INCLUDING the start error of the
episode (FORGE places the peg a few mm to ~3 cm from the hole). It does not measure the noise. Here, for every noisy demo,
the clean demo with the same seed (same start state, the reset is deterministic) gives the trajectory without noise, so
the difference of the two is the effect of the noise alone.

Columns (all differences noisy - clean, time-aligned step by step):
  noise   steps where noise was sent, and its mean norm (mrad, 7 joints) once the clip is applied
  dxy0    xy difference of the peg offset at t = 0 (mm): CONTROL, must be ~0 (same start state)
  dxy     largest xy difference of the peg offset (mm) from the first noisy step to 8 steps after the last one
  at      step of that maximum
  dq      largest joint-angle difference (mrad) in the same window
  +steps  steps of the noisy demo minus steps of the clean demo (what the noise costs in time)
"""
import argparse
from pathlib import Path

import numpy as np


def load(path):
    z = np.load(path)
    return {k: np.asarray(z[k]) for k in z.files}


def main():
    p = argparse.ArgumentParser(description="Effect of the injected noise on the peg: noisy demo vs the clean demo of the same seed.")
    p.add_argument("--noisy", required=True, help="Folder of the noisy demos (ep_*_seedS_noisy.npz).")
    p.add_argument("--clean", required=True, help="Folder of the clean demos of the SAME seeds (ep_*_seedS.npz).")
    p.add_argument("--tail", type=int, default=8, help="Steps after the last noisy step still included in the window.")
    args = p.parse_args()

    rows = []
    print(f"{'seed':>4} | {'noise':>5} {'|n| mrad':>8} | {'dxy0':>5} | {'dxy mm':>7} {'at':>3} | {'dq mrad':>7} | {'+steps':>6}")
    for f in sorted(Path(args.noisy).glob("ep_*_seed*_noisy.npz")):
        n = load(f)
        seed = int(n["seed"])
        cands = [c for c in Path(args.clean).glob(f"ep_*_seed{seed}.npz")]
        if not cands:
            print(f"{seed:4d} | no clean demo with this seed in {args.clean}")
            continue
        c = load(cands[0])
        T = min(len(n["peg_socket_offset"]), len(c["peg_socket_offset"]))
        noise = n["noise"][:T]
        on = np.flatnonzero(np.any(noise != 0.0, axis=1))
        if len(on) == 0:
            print(f"{seed:4d} | no noisy step")
            continue
        lo, hi = int(on[0]), min(int(on[-1]) + args.tail + 1, T)
        d_off = 1000.0 * np.linalg.norm((n["peg_socket_offset"][:T, :2] - c["peg_socket_offset"][:T, :2]), axis=1)
        d_q = 1000.0 * np.abs(n["joint_position"][:T] - c["joint_position"][:T]).max(axis=1)
        k = lo + int(np.argmax(d_off[lo:hi]))
        mean_norm = 1000.0 * float(np.linalg.norm(noise[on], axis=1).mean())
        rows.append((seed, d_off[0], d_off[k], d_q[lo:hi].max(), len(n["actions"]) - len(c["actions"])))
        print(f"{seed:4d} | {len(on):5d} {mean_norm:8.2f} | {d_off[0]:5.2f} | {d_off[k]:7.2f} {k:3d} | {d_q[lo:hi].max():7.2f} | "
              f"{len(n['actions']) - len(c['actions']):+6d}")
    if rows:
        a = np.array(rows, dtype=float)
        print(f"\n{len(rows)} demos: control dxy0 max {a[:, 1].max():.2f} mm | effect of the noise on the peg (dxy): "
              f"median {np.median(a[:, 2]):.2f} mm, max {a[:, 2].max():.2f} mm | extra steps: median {np.median(a[:, 4]):+.0f}")


if __name__ == "__main__":
    main()
