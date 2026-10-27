"""Does the gripper (not the peg) stop the insertion? Measures, at every reset, how far the peg sticks out below the fingers.

Client, runs in .venv, AFTER forge_server.py is up (restarted after the forge_server.py change that adds the debug op
`grasp_geometry`; the server decides the robot, `--set robot=...`). Nothing is moved: for each seed the simulation is reset
and the geometry is read.

    python scripts/isaaclab/grasp_check.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --num-seeds 10 --observed 0=3.5,5=8.2,8=3.74,9=4.39

Why. With the Robotiq the scripted demos go smoothly 17-21 mm into the 25 mm hole and then stop at a height that changes
from one seed to the next (3.5-8.2 mm above the final one), with 3-6 N on the wrist. If the fingers are lower than the
bottom of the peg's grip, they reach the socket top before the peg is fully in. For each seed the script prints

    grasp height   height of FORGE's fingertip midpoint above the peg base (where FORGE says the grasp is)
    h_min          lowest COLLISION vertex of any robot body above the peg base, among the vertices between
                   --lateral-min-mm and --lateral-max-mm from the peg axis (at least: they cannot go into the hole; at
                   most: leaves out the rest of the robot), and which body it belongs to. Visual-only meshes (listed with
                   "(visual)") have no collision: they would go through the socket
    predicted stall   hole_depth - h_min: the height of the peg base (mm above its final height) where that body reaches
                   the socket top. Negative = the fingers never touch the socket.
    peg tilt       angle between the peg axis and the vertical (a tilted peg jams in the hole)

--observed SEED=Z,... (mm: the final offset z of the failed attempts) adds the observed stall height to the table.
Reading it: predicted ~ observed for every seed -> the fingers hit the socket (the grasp is too deep for this gripper);
predicted ~ the same value for all seeds but observed varies -> something else varies; predicted negative -> not the fingers.
The vertices are the collision meshes of the USD when a body has some (column `vertices`), else its visible meshes.
Run it once with `--set robot=franka` and once with `--set robot=robotiq_2f85_tsf` to compare.

--out FILE writes everything (all seeds, all bodies listed) as JSON.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.env_client import EnvClient  # noqa: E402


def parse_observed(text):
    """'0=3.5,5=8.2' -> {0: 3.5, 5: 8.2}"""
    out = {}
    for item in filter(None, (s.strip() for s in (text or "").split(","))):
        seed, _, z = item.partition("=")
        out[int(seed)] = float(z)
    return out


def main():
    p = argparse.ArgumentParser(description="Peg protrusion below the fingers, per reset seed.")
    p.add_argument("--config", required=True, help="Task YAML (host, port).")
    p.add_argument("--seed", type=int, default=0, help="First reset seed.")
    p.add_argument("--num-seeds", type=int, default=10, help="Seeds seed ... seed + num-seeds - 1.")
    p.add_argument("--lateral-min-mm", type=float, default=5.0,
                   help="Only vertices at least this far from the peg axis count (a peg of 8 mm has a radius of 4 mm).")
    p.add_argument("--lateral-max-mm", type=float, default=60.0,
                   help="Only vertices at most this far from the peg axis count (leaves out the rest of the robot, e.g. its base).")
    p.add_argument("--top", type=int, default=6, help="Bodies listed for the first seed (the lowest ones).")
    p.add_argument("--observed", default="", help="Observed stall heights, mm: SEED=Z,SEED=Z,... (final offset z of failed demos).")
    p.add_argument("--out", default=None, help="Write all the results here as JSON.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    observed = parse_observed(args.observed)
    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, _ = client.create_env({})

    def debug(kind, **kw):
        res = client._call_operation("debug", {"env_id": env_id, "kind": kind, **kw})["result"]
        if isinstance(res, dict) and "error" in res:
            raise SystemExit(f"server: {res['error']}  (is forge_server.py the version with `grasp_geometry`? restart it after pulling)")
        return res

    results = []
    for i in range(args.num_seeds):
        seed = args.seed + i
        client.reset(env_id, seed)
        g = debug("grasp_geometry", lateral_min_mm=args.lateral_min_mm, lateral_max_mm=args.lateral_max_mm,
                  top=max(args.top, 8))
        g["seed"] = seed
        results.append(g)
        if i == 0:
            print(f"robot {g['robot']} | hole depth {g['hole_depth_mm']:.1f} mm | {g['bodies_with_vertices']}/{g['bodies']} "
                  f"bodies with vertices | vertices counted between {args.lateral_min_mm:g} and {args.lateral_max_mm:g} mm "
                  f"off the peg axis | h_min and the prediction use the COLLISION meshes")
            print(f"\nlowest bodies for seed {seed} (height above the peg base):")
            for r in g["lowest"][: args.top]:
                print(f"    {r['body']:34s} {r['h_mm']:7.1f} mm   {r['lateral_mm']:5.1f} mm off the axis   ({r['vertices']})")
            print("\nseed | grasp height | h_min | lowest body                    | predicted stall | observed | tilt")

    for g in results:
        obs = observed.get(g["seed"])
        pred = g["stall_z_pred_mm"]
        print(f"{g['seed']:4d} | {g['fingertip_midpoint_height_mm']:9.1f} mm | "
              f"{'-' if g['h_min_mm'] is None else format(g['h_min_mm'], '5.1f')} | {str(g['h_min_body']):30s} | "
              f"{'-' if pred is None else format(pred, '10.1f') + ' mm':>15s} | "
              f"{'-' if obs is None else format(obs, '5.1f') + ' mm':>8s} | {g['peg_tilt_deg']:4.1f} deg")

    preds = [g["stall_z_pred_mm"] for g in results if g["stall_z_pred_mm"] is not None]
    if preds:
        print(f"\npredicted stall height: min {min(preds):.1f} / mean {sum(preds) / len(preds):.1f} / max {max(preds):.1f} mm "
              f"(positive = the fingers hit the socket before the peg is fully in)")
    if observed:
        pairs = [(g["stall_z_pred_mm"], observed[g["seed"]]) for g in results
                 if g["seed"] in observed and g["stall_z_pred_mm"] is not None]
        if pairs:
            err = [abs(a - b) for a, b in pairs]
            print(f"observed seeds: mean |predicted - observed| = {sum(err) / len(err):.1f} mm over {len(pairs)} seeds")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
