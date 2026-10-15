"""Summarise the peg_socket_metrics.jsonl files written by the FORGE evaluations.

Each evaluation (scripts/eval_policy.py through IsaacLabEnvWrapper) writes one line per episode into
<video_dir>/peg_socket_metrics.jsonl: how close the held peg got to the hole and where it ended. This
script turns those files into one table per file / checkpoint, WITHOUT running anything on the cluster:

    python scripts/isaaclab/analyze_peg_metrics.py --paths logs/eval_videos

`--paths` takes files or folders (a folder is searched recursively for peg_socket_metrics.jsonl), so a whole
eval_curve output folder can be given at once. One row per file, labelled with its parent folder.

Why it exists: until IsaacLabEnvWrapper got `success_hold_steps`, the evaluation reported 0 % for any policy
(the success flag was read after the simulation had auto-reset, see that class). The old files have no
`success` field, so the success shown here is an APPROXIMATION from the END state of the episode:
    approx. success = final xy < --success-xy-mm  and  final z < --success-z-mm
(an insertion that was held and then lost is missed; a peg that only passes through is not counted). Give
several --success-z-mm values to see how sensitive the number is to the z threshold. New files carry the real
`success` field: it is shown in its own column (`real`, with the mean length of the successful episodes in
`steps|ok`) and it decides the `inserted` class.

Files written by the wrapper since it logs the contact force and the command gap also give, as medians over the
episodes of the last 20 steps of each: `F20` the contact force (N), `gap20` how far the commanded joint targets
lead the measured joints (mrad, the demos have about 6 inside the hole, see demo_z_bands.py) and `dz20` the z
change of the peg (mm; about 0 = stalled). Reading: F20 high = jammed; F20 and gap20 both ~0 and dz20 ~0 = the
policy stopped commanding motion.

Outcome of each episode (from the final state, mm; z = peg base height above its target pose, the rim of the
hole is at z = hole depth):
    inserted   approx. success
    lost       final xy >= --lost-xy-mm: the peg went away from the hole
    in_hole    below the rim (z < depth - 3) and xy < 5, not inserted: in the hole but not deep enough
    rim        |z - depth| within [-3, +8] and not lost: sitting on the edge of the hole
    high       above the rim by more than 8 mm and not lost: still hovering
    other      anything else
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

FILE_NAME = "peg_socket_metrics.jsonl"


def find_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.rglob(FILE_NAME))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"not found: {p}")
    if not files:
        sys.exit(f"no {FILE_NAME} found under: {' '.join(map(str, paths))}")
    # natural order: .../200 before .../1000
    return sorted(files, key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(f))])


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify(xy, z, args, z_tol):
    if xy < args.success_xy_mm and z < z_tol:
        return "inserted"
    if xy >= args.lost_xy_mm:
        return "lost"
    if z < args.hole_depth_mm - 3.0 and xy < 5.0:
        return "in_hole"
    if args.hole_depth_mm - 3.0 <= z <= args.hole_depth_mm + 8.0:
        return "rim"
    if z > args.hole_depth_mm + 8.0:
        return "high"
    return "other"


def summarise(records, args):
    xy = np.array([r["final_xy_mm"] for r in records])
    z = np.array([r["final_offset_mm"][2] for r in records])
    closest = np.array([r["closest_xy_mm"] for r in records])
    n = len(records)
    out = {"n": n, "median_final_xy": float(np.median(xy)), "median_closest_xy": float(np.median(closest)),
           "closest_below_1mm": int((closest < 1.0).sum())}
    for z_tol in args.success_z_mm:
        out[f"approx_success_z<{z_tol:g}"] = int(((xy < args.success_xy_mm) & (z < z_tol)).sum())
    z_tol = args.success_z_mm[0]
    classes = []
    for r, a, b in zip(records, xy, z):
        real = r.get("success")                       # new files: the real success decides "inserted"
        if real is True:
            classes.append("inserted")
            continue
        c = classify(a, b, args, z_tol)
        classes.append("in_hole" if (real is False and c == "inserted") else c)
    for name in ("inserted", "in_hole", "rim", "high", "lost", "other"):
        out[name] = classes.count(name)
    real = [r.get("success") for r in records]
    out["real_success"] = int(sum(bool(v) for v in real)) if all(v is not None for v in real) else None
    won = [r["steps"] for r in records if r.get("success")]
    out["steps_success"] = float(np.mean(won)) if won else None      # mean length of the successful episodes
    # contact force / command gap / z motion over the last 20 steps (files written after those fields were added)
    for key, field in (("force20", "force_last20_N"), ("gap20", "gap_last20_mrad"), ("dz20", "z_change_last20_mm")):
        vals = [r.get(field) for r in records]
        out[key] = float(np.median(vals)) if all(v is not None for v in vals) else None
    return out


def main():
    p = argparse.ArgumentParser(description="Summarise peg_socket_metrics.jsonl files.")
    p.add_argument("--paths", nargs="+", required=True,
                   help=f"{FILE_NAME} files and/or folders searched recursively for them.")
    p.add_argument("--success-xy-mm", type=float, default=2.5, help="FORGE xy tolerance (mm).")
    p.add_argument("--success-z-mm", type=float, nargs="+", default=[1.0, 3.0],
                   help="z thresholds (mm) tried for the approximate success; the first one also drives the "
                        "outcome classes. Look up the real one in the collector's start-up line "
                        "('success: xy<..., z<... mm') or use the default 1.0 (hole depth 25 mm x 0.04).")
    p.add_argument("--hole-depth-mm", type=float, default=25.0, help="Depth of the hole (rim height, mm).")
    p.add_argument("--lost-xy-mm", type=float, default=20.0, help="Final xy distance from which a peg counts as lost.")
    p.add_argument("--csv", default=None, help="Also write the table to this CSV file.")
    args = p.parse_args()

    rows = []
    for path in find_files(args.paths):
        records = load(path)
        if records:
            rows.append((path.parent.name, path, summarise(records, args)))
    if not rows:
        sys.exit("the files are empty")

    # two runs with the same checkpoint folder name (both "1000"): prefix with the run folder to tell them apart
    names = [label for label, _, _ in rows]
    rows = [((f"{path.parent.parent.name}/{label}" if names.count(label) > 1 else label), str(path), s)
            for label, path, s in rows]

    z_cols = [f"approx_success_z<{z:g}" for z in args.success_z_mm]
    header = ["label", "n"] + [f"ok z<{z:g}" for z in args.success_z_mm] + \
             ["real", "steps|ok", "inserted", "in_hole", "rim", "high", "lost", "other", "med.final xy", "med.closest xy", "closest<1mm",
              "med F20 (N)", "med gap20 (mrad)", "med dz20 (mm)"]
    table = []
    for label, _, s in rows:
        table.append([label, s["n"]] + [f"{s[c]}/{s['n']}" for c in z_cols] +
                     ["-" if s["real_success"] is None else f"{s['real_success']}/{s['n']}",
                      "-" if s["steps_success"] is None else f"{s['steps_success']:.0f}",
                      s["inserted"], s["in_hole"], s["rim"], s["high"], s["lost"], s["other"],
                      f"{s['median_final_xy']:.1f}", f"{s['median_closest_xy']:.1f}", s["closest_below_1mm"]] +
                     ["-" if s[k] is None else f"{s[k]:.2f}" for k in ("force20", "gap20", "dz20")])
    widths = [max(len(str(r[i])) for r in [header] + table) for i in range(len(header))]
    for r in [header] + table:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(f"\napprox. success = final xy < {args.success_xy_mm:g} mm and final z < the z threshold (end state only). "
          f"Classes use z < {args.success_z_mm[0]:g} mm for 'inserted', rim at z = {args.hole_depth_mm:g} mm.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
        print(f"table written to {args.csv}")


if __name__ == "__main__":
    main()
