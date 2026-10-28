"""Are the reset seeds reproducible? Same seed -> same start state and same physics, within one server run and across runs.

Client, runs in .venv, AFTER forge_server.py is up (same robot / YAML as the demos). Nothing moves except the arm holding its
pose, so it is quick (a few minutes). Do NOT run it while a collection uses the same server.

    # 1) inside ONE server run: each seed is reset --repeats times (the loop over the seeds is reversed on odd repeats, so
    #    the history before a reset is not the same); results of repeat 0 are saved
    python scripts/isaaclab/repro_check.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --seeds 0,1,2,3,13,40,87 --repeats 3 --save /tmp/repro_run1.json

    # 2) restart the server (same command as before), then compare with the first run
    python scripts/isaaclab/repro_check.py --config ... --seeds 0,1,2,3,13,40,87 --repeats 1 --compare /tmp/repro_run1.json

What is measured per (seed, repeat): the observation right after reset(seed) (joint angles, offset peg -> hole, finger opening,
finger effort, taxel sum), then --hold-steps steps holding the same joint target (the one of repeat 0, so the commands are
identical) and the same quantities at the end. The report gives the largest difference between repeats for each seed:

    reset q       rad,  joint angles after the reset      0 = same start state, bit for bit
    reset w       mm,   finger opening after the reset     the physics of the 0.25 s closing (GPU contacts)
    reset off     mm,   offset peg -> hole after the reset
    end off       mm,   offset peg -> hole after the hold
    end w         mm,   finger opening after the hold
    end tac       N,    sum of the taxels after the hold
    end eff       N.m,  finger motor effort after the hold

EXACT = every difference is 0.0 (bit for bit). The thresholds of "ok" (--tol-q 1e-6 rad, --tol-mm 0.01 mm, --tol-n 0.05 N) say
what is repeatable for practical purposes. For a contact-rich test of the insertion itself, use replay_demos.py: it replays a
recorded demo open loop and reports `drift` against the recording (made in another server run).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.env_client import EnvClient  # noqa: E402

QUANTITIES = ["reset_q", "reset_w", "reset_off", "end_off", "end_w", "end_tac", "end_eff"]
UNITS = {"reset_q": "rad", "reset_w": "mm", "reset_off": "mm", "end_off": "mm", "end_w": "mm", "end_tac": "N", "end_eff": "N.m"}


def snapshot(client, env_id, obs):
    """The numbers compared between repeats (lists of floats, JSON friendly)."""
    width = client._call_operation("debug", {"env_id": env_id, "kind": "finger_width"})["result"]
    tac = obs.get("extra/tactile")
    eff = obs.get("extra/gripper_effort")
    return {
        "q": np.asarray(obs["observation/joint_position"], dtype=np.float64).reshape(-1).tolist(),
        "off": (1000.0 * np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64)).reshape(-1).tolist(),
        "w": [1000.0 * float(width)] if not isinstance(width, dict) else [float("nan")],
        "tac": [0.0] if tac is None else [float(np.asarray(tac, dtype=np.float64).sum())],
        "eff": [0.0] if eff is None else [float(np.asarray(eff, dtype=np.float64).reshape(-1)[0])],
    }


def run_seed(client, env_id, seed, hold_steps, target):
    """Reset with `seed`, hold `target` (None = the joint angles of this very reset) and return both snapshots + the target."""
    obs, _ = client.reset(env_id, seed)
    first = snapshot(client, env_id, obs)
    if target is None:
        target = first["q"][:7]
    action = np.concatenate([np.asarray(target[:7], dtype=np.float64), [1.0]]).astype(np.float32)
    for _ in range(hold_steps):
        client.step(env_id, action)
        obs = client.get_observation(env_id)
    return {"reset": first, "end": snapshot(client, env_id, obs), "target": list(target[:7])}


def flat(rec):
    """{quantity: np.ndarray} of one (seed, repeat) record."""
    return {"reset_q": np.array(rec["reset"]["q"]), "reset_w": np.array(rec["reset"]["w"]), "reset_off": np.array(rec["reset"]["off"]),
            "end_off": np.array(rec["end"]["off"]), "end_w": np.array(rec["end"]["w"]), "end_tac": np.array(rec["end"]["tac"]),
            "end_eff": np.array(rec["end"]["eff"])}


def deviation(recs):
    """Largest absolute difference between the records of one seed, per quantity (nan if shapes differ)."""
    fl = [flat(r) for r in recs]
    out = {}
    for k in QUANTITIES:
        ref = fl[0][k]
        out[k] = max((float(np.abs(f[k] - ref).max()) if f[k].shape == ref.shape else float("nan")) for f in fl[1:])
    return out


def report(title, per_seed, tol):
    print(f"\n{title}")
    print("seed | " + " | ".join(f"{q + ' (' + UNITS[q] + ')':>16s}" for q in QUANTITIES) + " | verdict")
    n_exact = n_ok = 0
    for seed, dev in per_seed.items():
        exact = all(v == 0.0 for v in dev.values())
        ok = (dev["reset_q"] <= tol["q"] and dev["reset_w"] <= tol["mm"] and dev["reset_off"] <= tol["mm"] and dev["end_off"] <= tol["mm"]
              and dev["end_w"] <= tol["mm"] and dev["end_tac"] <= tol["n"] and dev["end_eff"] <= tol["n"])
        n_exact += exact
        n_ok += ok
        print(f"{seed:4d} | " + " | ".join(f"{dev[q]:16.3g}" for q in QUANTITIES) + f" | {'EXACT' if exact else ('ok' if ok else 'DIFFERENT')}")
    print(f"{len(per_seed)} seeds: {n_exact} exact, {n_ok} within tolerance (q {tol['q']:g} rad, {tol['mm']:g} mm, {tol['n']:g} N)")


def main():
    p = argparse.ArgumentParser(description="Reproducibility of reset seeds (within one server run and across server runs).")
    p.add_argument("--config", required=True, help="Task YAML (host, port).")
    p.add_argument("--seeds", default="0,1,2,3,13,40,87", help="Comma list of reset seeds.")
    p.add_argument("--repeats", type=int, default=3, help="How many times each seed is reset in this run.")
    p.add_argument("--hold-steps", type=int, default=30, help="Steps holding the pose after each reset.")
    p.add_argument("--save", default=None, help="Write the records of this run (JSON) to compare a later server run with.")
    p.add_argument("--compare", default=None, help="JSON saved by a previous server run: compared with repeat 0 of this run.")
    p.add_argument("--tol-q", type=float, default=1e-6)
    p.add_argument("--tol-mm", type=float, default=0.01)
    p.add_argument("--tol-n", type=float, default=0.05)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tol = {"q": args.tol_q, "mm": args.tol_mm, "n": args.tol_n}
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, _ = client.create_env({})

    records = {s: [] for s in seeds}                          # seed -> [record of repeat 0, 1, ...]
    targets = {}
    for rep in range(args.repeats):
        order = seeds if rep % 2 == 0 else seeds[::-1]
        for seed in order:
            rec = run_seed(client, env_id, seed, args.hold_steps, targets.get(seed))
            targets.setdefault(seed, rec["target"])
            records[seed].append(rec)
        print(f"repeat {rep} done", flush=True)

    if args.repeats > 1:
        report(f"Within this server run: {args.repeats} resets of each seed (largest difference between repeats)",
               {s: deviation(r) for s, r in records.items()}, tol)

    if args.save:
        Path(args.save).write_text(json.dumps({str(s): r for s, r in records.items()}))
        print(f"\nsaved: {args.save}")

    if args.compare:
        old = json.loads(Path(args.compare).read_text())
        per_seed = {}
        for s in seeds:
            if str(s) in old:
                # same hold target in both runs: the one of the old run, so only the simulation can differ at the end
                per_seed[s] = deviation([old[str(s)][0], records[s][0]])
        report(f"Across server runs: repeat 0 of {args.compare} vs repeat 0 of this run", per_seed, tol)
        print("(the hold target of the new run is its own reset pose: a difference in `reset q` also shifts the end values)")


if __name__ == "__main__":
    main()
