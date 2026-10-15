"""Normalization statistics computed on the FORGE waypoint demos, in openpi's norm_stats.json format.

    python scripts/isaaclab/compute_demo_norm_stats.py \
        --demos-dir demos/isaaclab/ForgePegInsert \
        --output-dir assets/forge_demo_norm \
        --asset-id forge_demos

Writes <output-dir>/<asset-id>/norm_stats.json, ready for the SFT (sft_norm_stats_dir / sft_norm_stats_asset_id
in configs/task/isaaclab/peg_insert_forge_sft_demo_norm.yaml). No GPU, no lerobot: only numpy, reads the
joint_position / gripper_position / actions arrays of the ep_*.npz files (not the images).

What is computed (exactly what openpi's compute_norm_stats.py would see with the "..._joint_state_delta" config):
    state    (N, 8)      joint_position (7) + gripper_position (1), one row per frame
    actions  (N*16, 8)   the 16-step chunk of every frame as OFFSETS from the joint state at the chunk start
                         (joints) + the absolute gripper command; the last action repeated past the end of the
                         episode (chunk_offsets of convert_waypoint_demos_to_lerobot.py, same function)
and per dimension mean, std, q01, q99 (pi05 normalizes with the quantiles: (x - q01) / (q99 - q01) * 2 - 1).

Two safeguards, because a dimension that hardly moves in 50 scripted demos gets a tiny range and the
normalization divides by it, which amplifies any noise (and by ~1e-6 for a constant dimension):
    --min-range-frac F   the range q99 - q01 (and the std) of a joint dimension is at least F times the DROID one
                         (default 0.02; 0 disables). The range is widened around its centre.
    gripper (dimension 8) the DROID statistics are kept (the gripper stays closed in FORGE: constant in the demos,
                         so its demo statistics would be degenerate). --demo-gripper uses the demo ones instead.
Both need the official DROID norm_stats.json (--droid-stats; the default is the path used on the cluster by the
conversion script). Without that file only the raw demo statistics can be written: pass --no-droid.

The table printed at the end compares the demo ranges with the DROID ones: ratio demo/DROID well below 1 means the
normalized values get stretched by 1/ratio compared with what the pretrained model saw.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_waypoint_demos_to_lerobot import ACTION_HORIZON, DEFAULT_NORM_STATS, chunk_offsets   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = ("state", "actions")
FIELDS = ("mean", "std", "q01", "q99")


def read_demos(files):
    """state (N, 8) and chunk offsets (N * horizon, 8) of all the demos."""
    states, chunks = [], []
    for f in files:
        d = np.load(f)
        for k in ("joint_position", "gripper_position", "actions"):
            if k not in d.files:
                raise SystemExit(f"{f.name}: missing array {k!r}")
        q = d["joint_position"].astype(np.float32)
        grip = d["gripper_position"].astype(np.float32).reshape(-1, 1)
        act = d["actions"].astype(np.float32)
        if q.shape != (len(act), 7) or act.shape[1] != 8 or len(grip) != len(act):
            raise SystemExit(f"{f.name}: unexpected shapes q {q.shape}, gripper {grip.shape}, actions {act.shape}")
        states.append(np.concatenate([q, grip], axis=1))
        chunks.append(chunk_offsets(q, act, ACTION_HORIZON).reshape(-1, 8))
    return np.concatenate(states).astype(np.float64), np.concatenate(chunks).astype(np.float64)


def stats_of(x):
    return {"mean": x.mean(0), "std": x.std(0), "q01": np.quantile(x, 0.01, axis=0), "q99": np.quantile(x, 0.99, axis=0)}


def load_droid(path):
    ns = json.load(open(path))["norm_stats"]
    return {k: {f: np.asarray(ns[k][f], dtype=np.float64)[:8] for f in FIELDS} for k in KEYS}


def apply_safeguards(demo, droid, frac, keep_droid_gripper):
    """Returns (final stats, floored): floored[key] = list of the dimensions whose range was widened."""
    out, floored = {}, {}
    for key in KEYS:
        s = {f: demo[key][f].copy() for f in FIELDS}
        floored[key] = []
        if droid is not None:
            for i in range(7):
                min_range = frac * (droid[key]["q99"][i] - droid[key]["q01"][i])
                rng = s["q99"][i] - s["q01"][i]
                if rng < min_range:
                    centre = 0.5 * (s["q99"][i] + s["q01"][i])
                    s["q01"][i], s["q99"][i] = centre - 0.5 * min_range, centre + 0.5 * min_range
                    floored[key].append(i)
                s["std"][i] = max(s["std"][i], frac * droid[key]["std"][i])
            if keep_droid_gripper:
                for f in FIELDS:
                    s[f][7] = droid[key][f][7]
        out[key] = s
    return out, floored


def main():
    p = argparse.ArgumentParser(description="Demo normalization statistics in openpi norm_stats.json format.")
    p.add_argument("--demos-dir", required=True, help="Folder with the ep_*.npz files of collect_waypoint_demos.py.")
    p.add_argument("--output-dir", required=True,
                   help="Assets folder: the file is written to <output-dir>/<asset-id>/norm_stats.json. "
                        "Relative paths are relative to the repo root.")
    p.add_argument("--asset-id", default="forge_demos", help="Sub-folder name (openpi 'asset id').")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Use only the first N demos (sorted by name). Give the SFT's num_data_sft (the dataset "
                        "conversion keeps the same order). Default: all.")
    p.add_argument("--droid-stats", default=DEFAULT_NORM_STATS,
                   help="Official DROID norm_stats.json (pi05_droid_jointpos assets), for the safeguards and the table.")
    p.add_argument("--no-droid", action="store_true",
                   help="Do not use the DROID statistics: raw demo statistics, no safeguards, no comparison.")
    p.add_argument("--min-range-frac", type=float, default=0.02,
                   help="Minimum range / std of a joint dimension, as a fraction of the DROID one (0 disables).")
    p.add_argument("--demo-gripper", action="store_true",
                   help="Use the demo statistics for the gripper dimension too (default: keep the DROID ones).")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing norm_stats.json.")
    args = p.parse_args()

    demos_dir = Path(args.demos_dir)
    if not demos_dir.is_absolute():
        demos_dir = REPO_ROOT / demos_dir
    files = sorted(demos_dir.glob("ep_*.npz"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        sys.exit(f"no ep_*.npz in {demos_dir}")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_file = out_dir / args.asset_id / "norm_stats.json"
    if out_file.exists() and not args.overwrite:
        sys.exit(f"{out_file} already exists: pass --overwrite to replace it.")

    droid = None
    if not args.no_droid:
        if not Path(args.droid_stats).exists():
            sys.exit(f"DROID norm_stats.json not found at {args.droid_stats}: pass --droid-stats PATH "
                     "(or --no-droid for the raw demo statistics, without safeguards).")
        droid = load_droid(args.droid_stats)

    state, chunks = read_demos(files)
    print(f"{len(files)} demos from {demos_dir}: {len(state)} frames, {len(chunks)} chunk rows (horizon {ACTION_HORIZON})")
    demo = {"state": stats_of(state), "actions": stats_of(chunks)}
    final, floored = apply_safeguards(demo, droid, args.min_range_frac, not args.demo_gripper)

    np.set_printoptions(precision=4, suppress=True, linewidth=170)
    names = [f"joint{i + 1}" for i in range(7)] + ["gripper"]
    for key in KEYS:
        print(f"\n== {key}: quantile range q99 - q01 (rad; gripper 0-1)")
        print(f"{'dim':8s} {'demo q01':>10s} {'demo q99':>10s} {'demo range':>11s}", end="")
        print(f" {'DROID range':>12s} {'ratio':>8s}  final range  note" if droid is not None else "")
        for i, n in enumerate(names):
            d_rng = demo[key]["q99"][i] - demo[key]["q01"][i]
            f_rng = final[key]["q99"][i] - final[key]["q01"][i]
            line = f"{n:8s} {demo[key]['q01'][i]:10.4f} {demo[key]['q99'][i]:10.4f} {d_rng:11.4f}"
            if droid is not None:
                r_rng = droid[key]["q99"][i] - droid[key]["q01"][i]
                note = ("widened to the floor" if i in floored[key]
                        else "DROID kept" if (i == 7 and not args.demo_gripper) else "")
                line += f" {r_rng:12.4f} {d_rng / r_rng if r_rng else float('nan'):8.3f}  {f_rng:11.4f}  {note}"
            print(line)

    n_state = (state - final["state"]["q01"]) / (final["state"]["q99"] - final["state"]["q01"] + 1e-6) * 2 - 1
    n_act = (chunks - final["actions"]["q01"]) / (final["actions"]["q99"] - final["actions"]["q01"] + 1e-6) * 2 - 1
    print("\nwhat the model sees with these statistics (demos, quantile-normalized)")
    print("  state   share outside [-1, 1] per dim:", (np.abs(n_state) > 1).mean(0))
    print("  actions std per dim                   :", n_act.std(0))
    print("  actions min / max (joints)            :", n_act[:, :7].min(), "/", n_act[:, :7].max())

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"norm_stats": {k: {f: final[k][f].tolist() for f in FIELDS} for k in KEYS}}
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {out_file}")


if __name__ == "__main__":
    main()
