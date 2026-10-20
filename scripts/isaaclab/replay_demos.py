"""Replay recorded demos (.npz of collect_waypoint_demos.py / teleop_client.py) in the simulation, to check them.

Runs in .venv (client side, like collect_waypoint_demos.py), AFTER forge_server.py is up, started from the SAME task YAML
as the one the demos were recorded with:

    python scripts/isaaclab/replay_demos.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --demos-dir demos/isaaclab/ForgePegInsertTeleop

For every ep_*.npz: reset with the demo's seed, send its 8-value actions one by one (open loop: nothing is corrected),
and look at what FORGE says. The recorded frame t is the observation BEFORE action t, so the offset peg -> hole after
action t must be the recorded offset of frame t + 1. What is checked:

    reset     the observation after reset(seed) is the first recorded frame (joint angles within 1e-4 rad):
              the same seed gives the same start state
    success   FORGE says "inserted" for HOLD_AFTER_SUCCESS steps in a row at the end of the replay (that is how the demo
              was ended when it was recorded)
    drift     largest distance (mm) between the replayed and the recorded offset peg -> hole over the whole demo:
              ~0 = the simulation reproduces the demo, large = the physics (contacts) is not repeatable

OK = replayed with success. WARN = success but the replay drifted more than --drift-warn-mm from the recording. FAIL = no
success. A FAIL does NOT necessarily mean a bad demo: the replay needs a repeatable simulation, the policy does not. Look
at `drift` and at where it starts (printed: first step beyond the threshold) before throwing a demo away.
The report is also written to <demos-dir>/replay_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.env_client import EnvClient  # noqa: E402

HOLD_AFTER_SUCCESS = 5    # same as the collectors: success = "inserted" this many steps in a row


def replay_one(client, env_id, path, drift_warn_mm):
    z = np.load(path)
    seed = int(z["seed"])
    actions = np.asarray(z["actions"], dtype=np.float32)
    recorded = np.asarray(z["peg_socket_offset"], dtype=np.float64)          # (T, 3), m: frame t = before action t
    joints0 = np.asarray(z["joint_position"][0], dtype=np.float64)
    steps = len(actions)

    obs, _ = client.reset(env_id, seed)
    reset_dev = float(np.abs(np.asarray(obs["observation/joint_position"], dtype=np.float64) - joints0).max())

    in_a_row, best_run, first_success, ended_early = 0, 0, None, False
    drift = []                                                               # mm, replayed vs recorded, frames 1 .. T-1
    final_offset = None
    for t in range(steps):
        client.step(env_id, actions[t])
        obs = client.get_observation(env_id)
        done, inserted, _, _ = client.get_info_for_step(env_id)
        if done:                                                             # timeout: the env has already reset itself
            ended_early = True
            break
        offset = np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64)
        final_offset = offset
        if t + 1 < steps:
            drift.append(float(np.linalg.norm(offset - recorded[t + 1]) * 1000.0))
        in_a_row = in_a_row + 1 if inserted else 0
        best_run = max(best_run, in_a_row)
        if inserted and first_success is None:
            first_success = t
    success = best_run >= HOLD_AFTER_SUCCESS and not ended_early
    drift_max = max(drift) if drift else 0.0
    beyond = next((i + 1 for i, d in enumerate(drift) if d > drift_warn_mm), None)
    verdict = ("WARN" if drift_max > drift_warn_mm else "OK") if success else "FAIL"
    return {
        "file": Path(path).name, "seed": seed, "steps": steps, "verdict": verdict, "success": bool(success),
        "longest_success_run": int(best_run), "first_success_step": first_success, "ended_early": ended_early,
        "reset_joint_dev_rad": reset_dev, "drift_max_mm": round(drift_max, 3),
        "drift_mean_mm": round(float(np.mean(drift)), 3) if drift else 0.0,
        "first_step_beyond_threshold": beyond,
        "final_offset_mm": None if final_offset is None else [round(float(v) * 1000.0, 2) for v in final_offset],
    }


def main():
    p = argparse.ArgumentParser(description="Replay recorded demos in the simulation and check that they succeed.")
    p.add_argument("--config", required=True, help="Task YAML of the running server (host, port).")
    p.add_argument("--demos-dir", default="demos/isaaclab/ForgePegInsertTeleop", help="Folder with ep_*.npz.")
    p.add_argument("--limit", type=int, default=0, help="Replay only the first N demos (0 = all).")
    p.add_argument("--drift-warn-mm", type=float, default=2.0,
                   help="WARN when the replayed offset peg -> hole differs from the recorded one by more than this.")
    p.add_argument("--report", default=None, help="Where to write the JSON report (default <demos-dir>/replay_report.json).")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    files = sorted(Path(args.demos_dir).glob("ep_*.npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no ep_*.npz in {args.demos_dir}")

    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, _ = client.create_env({})
    print(f"replaying {len(files)} demos from {args.demos_dir}  (open loop, success = {HOLD_AFTER_SUCCESS} 'inserted' "
          f"steps in a row, WARN beyond {args.drift_warn_mm:g} mm of drift)\n")
    print(f"{'file':28s} {'seed':>5s} {'steps':>5s}  {'verdict':7s} {'run':>3s}  {'drift max/mean (mm)':>20s}  "
          f"{'reset dev':>9s}  final offset (mm)")
    results = []
    t0 = time.time()
    for i, path in enumerate(files):
        r = replay_one(client, env_id, path, args.drift_warn_mm)
        results.append(r)
        extra = f"   drift > {args.drift_warn_mm:g} mm from step {r['first_step_beyond_threshold']}" \
            if r["first_step_beyond_threshold"] else ""
        print(f"{r['file']:28s} {r['seed']:5d} {r['steps']:5d}  {r['verdict']:7s} {r['longest_success_run']:3d}  "
              f"{r['drift_max_mm']:9.2f} / {r['drift_mean_mm']:<8.2f}  {r['reset_joint_dev_rad']:9.1e}  "
              f"{r['final_offset_mm']}{extra}", flush=True)

    counts = {v: sum(r["verdict"] == v for r in results) for v in ("OK", "WARN", "FAIL")}
    print(f"\n{len(results)} demos replayed in {time.time() - t0:.0f} s:  OK {counts['OK']}   WARN {counts['WARN']}   "
          f"FAIL {counts['FAIL']}    (success {counts['OK'] + counts['WARN']}/{len(results)})")
    worst_reset = max(r["reset_joint_dev_rad"] for r in results)
    print(f"reset reproducibility: worst joint deviation after reset(seed) = {worst_reset:.1e} rad "
          f"({'identical start state' if worst_reset < 1e-4 else 'START STATE DIFFERS: is the server run from the same YAML / CUBLAS_WORKSPACE_CONFIG=:4096:8 ?'})")
    failed = [r["file"] for r in results if r["verdict"] == "FAIL"]
    if failed:
        print("failed: " + ", ".join(failed))
    report = Path(args.report) if args.report else Path(args.demos_dir) / "replay_report.json"
    report.write_text(json.dumps({"counts": counts, "demos": results}, indent=1))
    print(f"report: {report}")


if __name__ == "__main__":
    main()
