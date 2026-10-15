"""Profile of the waypoint demos by height of the peg above its target pose (z of extra/peg_socket_offset).

    python scripts/isaaclab/demo_z_bands.py --demos-dir demos/isaaclab/ForgePegInsert

For each z band (rim of the hole at z = 25 mm) it prints, averaged over the demos:
  steps/demo               how many control steps a demo spends in the band
  target-state gap (mrad)  largest |commanded absolute joint target - measured joint position| over the 7 joints:
                           how far the command leads the arm, i.e. how hard the demo pushes in that band
  |F| (N)                  norm of the contact force (first 3 components of extra/ft_force)
Used to compare with where the SFT policy stalls (z ~ 14-18 mm, xy < 1 mm).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

BANDS = [("above rim  z>28", 28, 1e9), ("rim 22-28", 22, 28), ("mid  12-22", 12, 22),
         ("low  1-12", 1, 12), ("done  <1", -1e9, 1)]


def main():
    p = argparse.ArgumentParser(description="Time, command lead and contact force of the demos by z band.")
    p.add_argument("--demos-dir", required=True, help="Folder with the ep_*.npz files written by collect_waypoint_demos.py.")
    p.add_argument("--pattern", default="ep_*.npz", help="File name pattern inside --demos-dir.")
    args = p.parse_args()

    files = sorted(glob.glob(str(Path(args.demos_dir) / args.pattern)))
    if not files:
        sys.exit(f"no {args.pattern} in {args.demos_dir}  (try: ls -d demos/isaaclab/*/)")

    rows = {name: {"steps": [], "err": [], "force": []} for name, _, _ in BANDS}
    for f in files:
        d = np.load(f)
        z = d["peg_socket_offset"][:, 2] * 1000
        err = np.abs(d["actions"][:, :7] - d["joint_position"]).max(1) * 1000
        force = np.linalg.norm(d["ft_force"][:, :3], axis=1)
        for name, lo, hi in BANDS:
            m = (z >= lo) & (z < hi)
            rows[name]["steps"].append(int(m.sum()))
            if m.any():
                rows[name]["err"].append(err[m].mean())
                rows[name]["force"].append(force[m].mean())

    def avg(v):
        return float(np.mean(v)) if v else float("nan")

    print(f"{len(files)} demos from {args.demos_dir}")
    print(f"{'z band (mm)':18s} {'steps/demo':>10s} {'target-state gap (mrad)':>24s} {'|F| (N)':>9s}")
    for name, _, _ in BANDS:
        r = rows[name]
        print(f"{name:18s} {avg(r['steps']):10.1f} {avg(r['err']):24.2f} {avg(r['force']):9.2f}")


if __name__ == "__main__":
    main()
