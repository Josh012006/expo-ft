"""Statistics of a grip_check.py output file (the "reset trace" table), e.g. to compare two server settings.

    python scripts/isaaclab/grip_check.py --config ... --num-seeds 100 --hold-steps 3 > /tmp/grip100.txt
    python scripts/isaaclab/grip_stats.py /tmp/grip100.txt

A seed counts as "closed" when the fingers end the reset closing below --closed-mm (default 1 mm): fully closed instead of
stopped by the peg. Printed: how many, which ones, and for the others the spread of the placement offset along the pad
(|x|), of the peg offset across the closing axis after closing, of the finger opening after closing (w), of the sideways
shift of the peg during the closing and (if the file has it) of the finger-opening jitter over the last 15 steps. Parses only the second table of grip_check.py ("reset trace ...").
"""
import argparse
import re

import numpy as np

ROW = re.compile(r"^\s*(\d+) \| \(([^)]*)\) \| \(([^)]*)\)\s+(\S+) \| \(([^)]*)\)\s+(\S+) \| \s*(\S+) -> \s*(\S+)")


def spread(v):
    return "min %.2f  p10 %.2f  p50 %.2f  p90 %.2f  max %.2f" % (
        v.min(), np.percentile(v, 10), np.median(v), np.percentile(v, 90), v.max()) if len(v) else "-"


def main():
    p = argparse.ArgumentParser(description="Statistics of a grip_check.py output file.")
    p.add_argument("file", help="Output of grip_check.py (any run with the reset trace).")
    p.add_argument("--closed-mm", type=float, default=1.0, help="Finger opening after closing below which a seed is 'closed'.")
    args = p.parse_args()

    rows = []
    for line in open(args.file):
        m = ROW.match(line)
        if m:
            placed = [float(v) for v in m.group(3).split(",")]
            closed = [float(v) for v in m.group(5).split(",")]
            jit = re.search(r"jit\s+(\S+)", line)
            rows.append((int(m.group(1)), placed[0], closed[0], float(m.group(6)), float(m.group(8)),
                         float(jit.group(1)) if jit else np.nan))
    if not rows:
        raise SystemExit("no reset-trace rows found: is it the output of grip_check.py without --no-trace?")
    a = np.array(rows)                    # seed, x placed, x closed, across closed, w closed, jitter (nan if absent)
    bad = a[:, 4] < args.closed_mm
    ok = ~bad
    print(f"{len(a)} seeds, {int(bad.sum())} closed below {args.closed_mm:g} mm: {a[bad, 0].astype(int).tolist()}")
    print("|x| at placement, others :", spread(np.abs(a[ok, 1])))
    print("|x| at placement, closed :", spread(np.abs(a[bad, 1])))
    print("|across| after closing, others (mm):", spread(np.abs(a[ok, 3])))
    print("w after closing, others (mm)        :", spread(a[ok, 4]))
    print("sideways shift |x closed - x placed|, others (mm):", spread(np.abs(a[ok, 2] - a[ok, 1])))
    if not np.isnan(a[:, 5]).all():
        print("w jitter over the last 15 steps, others (mm)      :", spread(a[ok, 5]),
              f"| seeds above 0.10: {int((a[ok, 5] > 0.10).sum())} of {int(ok.sum())}")
    if bad.any():
        thr = np.abs(a[bad, 1]).min()
        print(f"others with |x placed| >= {thr:.2f} (the smallest of the closed ones): {int((np.abs(a[ok, 1]) >= thr).sum())} of {int(ok.sum())}")


if __name__ == "__main__":
    main()
