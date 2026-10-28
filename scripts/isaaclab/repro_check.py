"""Are the reset seeds reproducible? Same seed -> same start state and same physics, within one server run and across runs.

Client, runs in .venv, AFTER forge_server.py is up (same robot / YAML as the demos). Do NOT run it while a collection uses the
same server. Two tests, both repeated --repeats times for every item:

  hold    (--seeds)      reset(seed), then --hold-steps steps holding the same joint target. Tests the RESET: the random
                         placement, the 0.25 s closing of the gripper on the peg.
  replay  (--demos-dir)  reset(seed of a recorded demo), then its recorded actions sent open loop (nothing is corrected).
                         Tests the whole episode, contacts of the insertion included. For each pair of repeats it gives the
                         FIRST STEP where the arm or the peg differ at all (bit for bit), which says where the runs separate:
                         step 0 = the reset itself, a later step = the physics of the episode.

    # 1) inside ONE server run
    python scripts/isaaclab/repro_check.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --seeds 0,1,2,3,13,40,87 --demos-dir /tmp/clean_demos --demos-limit 3 --repeats 3 --save /tmp/repro_run1.json

    # 2) restart the server (same command as before), then compare with the first run
    python scripts/isaaclab/repro_check.py --config ... --seeds 0,1,2,3,13,40,87 --demos-dir /tmp/clean_demos --demos-limit 3 \
        --repeats 1 --compare /tmp/repro_run1.json

--order interleaved (default): repeat 0 goes through all the items, repeat 1 through them in reverse order, ... so the episode
that ran just before a reset is not the same from one repeat to the next. --order back-to-back: the repeats of an item follow
each other (same previous episode every time). Comparing the two says whether the history of the server matters.

Quantities per (seed, repeat) of the hold test: joint angles after the reset (rad), offset peg -> hole after the reset and after
the hold (mm), finger opening (mm), taxel sum (N), finger effort (N.m). The report gives the largest difference between repeats.
EXACT = every difference is 0.0 (bit for bit); "ok" = within --tol-q 1e-6 rad, --tol-mm 0.01 mm, --tol-n 0.05 N.
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


# ---------------------------------------------------------------------------------------------- hold test
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


def run_hold(client, env_id, seed, hold_steps, target):
    """Reset with `seed`, hold `target` (None = the joint angles of this very reset); both snapshots + the target."""
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
    return {"reset_q": np.array(rec["reset"]["q"]), "reset_w": np.array(rec["reset"]["w"]), "reset_off": np.array(rec["reset"]["off"]),
            "end_off": np.array(rec["end"]["off"]), "end_w": np.array(rec["end"]["w"]), "end_tac": np.array(rec["end"]["tac"]),
            "end_eff": np.array(rec["end"]["eff"])}


def deviation(recs):
    """Largest absolute difference between the records of one seed and the first one, per quantity."""
    fl = [flat(r) for r in recs]
    out = {}
    for k in QUANTITIES:
        ref = fl[0][k]
        out[k] = max((float(np.abs(f[k] - ref).max()) if f[k].shape == ref.shape else float("nan")) for f in fl[1:])
    return out


def report_hold(title, per_seed, tol):
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


# --------------------------------------------------------------------------------------------- replay test
def load_demos(demos_dir, limit):
    """[(name, seed, actions)] of the first `limit` demos (sorted by name) of a collection folder."""
    out = []
    for path in sorted(Path(demos_dir).glob("*.npz")):
        z = np.load(path)
        if "actions" not in z or "seed" not in z:
            continue
        out.append((path.name, int(z["seed"]), np.asarray(z["actions"], dtype=np.float32)))
        if len(out) >= limit:
            break
    return out


def run_replay(client, env_id, seed, actions):
    """Reset with `seed`, send `actions` open loop. Returns {"q": (T+1, 7) rad, "off": (T+1, 3) mm}; stops if the env says done."""
    obs, _ = client.reset(env_id, seed)
    q = [np.asarray(obs["observation/joint_position"], dtype=np.float64)[:7]]
    off = [1000.0 * np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64)]
    for a in actions:
        client.step(env_id, a)
        obs = client.get_observation(env_id)
        done, _, _, _ = client.get_info_for_step(env_id)
        if done:
            break
        q.append(np.asarray(obs["observation/joint_position"], dtype=np.float64)[:7])
        off.append(1000.0 * np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64))
    return {"q": np.stack(q).tolist(), "off": np.stack(off).tolist()}


def compare_traj(ref, other):
    """Largest differences and FIRST step where `other` differs at all (bit for bit) from `ref`."""
    n = min(len(ref["q"]), len(other["q"]))
    dq = np.abs(np.asarray(ref["q"][:n]) - np.asarray(other["q"][:n])).max(axis=1)          # per step
    do = np.abs(np.asarray(ref["off"][:n]) - np.asarray(other["off"][:n])).max(axis=1)
    differs = (dq > 0) | (do > 0)
    onset = int(np.argmax(differs)) if differs.any() else None
    return {"max_dq": float(dq.max()), "max_doff": float(do.max()), "onset": onset,
            "off_at_onset": None if onset is None else [round(v, 3) for v in ref["off"][onset]],
            "len": [len(ref["q"]), len(other["q"])]}


def report_replay(title, rows):
    print(f"\n{title}")
    print("demo                          | seed | steps | repeat | max dq (rad) | max doff (mm) | 1st different step | offset there (mm)")
    for name, seed, k, c in rows:
        onset = "identical" if c["onset"] is None else str(c["onset"])
        print(f"{name:29s} | {seed:4d} | {c['len'][0]:5d} | {k:6d} | {c['max_dq']:12.3g} | {c['max_doff']:13.3g} | {onset:>18s} | "
              f"{'-' if c['off_at_onset'] is None else c['off_at_onset']}")
    n_same = sum(1 for _, _, _, c in rows if c["onset"] is None)
    print(f"{len(rows)} comparisons: {n_same} bit for bit identical")


# -------------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Reproducibility of reset seeds and of open-loop episodes (within one server run and across runs).")
    p.add_argument("--config", required=True, help="Task YAML (host, port).")
    p.add_argument("--seeds", default="0,1,2,3,13,40,87", help="Comma list of reset seeds of the hold test ('' = skip it).")
    p.add_argument("--demos-dir", default=None, help="Folder of .npz demos (collect_waypoint_demos.py): open-loop replay test.")
    p.add_argument("--demos-limit", type=int, default=3, help="How many demos of the folder (first ones by name).")
    p.add_argument("--repeats", type=int, default=3, help="How many times each item is run in this server run.")
    p.add_argument("--order", choices=["interleaved", "back-to-back"], default="interleaved",
                   help="interleaved: all items, then all items reversed, ...; back-to-back: the repeats of an item follow each other.")
    p.add_argument("--hold-steps", type=int, default=30, help="Steps holding the pose after each reset (hold test).")
    p.add_argument("--save", default=None, help="Write the records of this run (JSON), to compare a later server run with.")
    p.add_argument("--compare", default=None, help="JSON saved by a previous server run: compared with repeat 0 of this run.")
    p.add_argument("--tol-q", type=float, default=1e-6)
    p.add_argument("--tol-mm", type=float, default=0.01)
    p.add_argument("--tol-n", type=float, default=0.05)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    demos = load_demos(args.demos_dir, args.demos_limit) if args.demos_dir else []
    if not seeds and not demos:
        raise SystemExit("nothing to do: give --seeds and / or --demos-dir (with .npz demos that have `actions` and `seed`)")
    tol = {"q": args.tol_q, "mm": args.tol_mm, "n": args.tol_n}
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, _ = client.create_env({})

    items = [("hold", s) for s in seeds] + [("replay", d[0]) for d in demos]
    by_name = {d[0]: d for d in demos}
    hold = {s: [] for s in seeds}                              # seed -> [record of repeat 0, 1, ...]
    replay = {d[0]: [] for d in demos}                         # demo name -> [trajectory of repeat 0, 1, ...]
    targets = {}

    def run_item(kind, key):
        if kind == "hold":
            rec = run_hold(client, env_id, key, args.hold_steps, targets.get(key))
            targets.setdefault(key, rec["target"])
            hold[key].append(rec)
        else:
            _, seed, actions = by_name[key]
            replay[key].append(run_replay(client, env_id, seed, actions))

    if args.order == "interleaved":
        for rep in range(args.repeats):
            for kind, key in (items if rep % 2 == 0 else items[::-1]):
                run_item(kind, key)
            print(f"repeat {rep} done", flush=True)
    else:
        for kind, key in items:
            for _ in range(args.repeats):
                run_item(kind, key)
            print(f"{kind} {key} done", flush=True)

    where = f"{args.repeats} runs of each item, order {args.order}"
    if args.repeats > 1:
        if seeds:
            report_hold(f"HOLD test within this server run ({where}; largest difference between repeats)",
                        {s: deviation(r) for s, r in hold.items()}, tol)
        if demos:
            rows = [(name, by_name[name][1], k, compare_traj(trs[0], trs[k])) for name, trs in replay.items() for k in range(1, len(trs))]
            report_replay(f"REPLAY test within this server run ({where}; each repeat vs repeat 0)", rows)

    if args.save:
        Path(args.save).write_text(json.dumps({"hold": {str(s): r for s, r in hold.items()}, "replay": replay}))
        print(f"\nsaved: {args.save}")

    if args.compare:
        old = json.loads(Path(args.compare).read_text())
        old_hold, old_replay = old.get("hold", {}), old.get("replay", {})
        per_seed = {s: deviation([old_hold[str(s)][0], hold[s][0]]) for s in seeds if str(s) in old_hold}
        if per_seed:
            report_hold(f"HOLD test across server runs: repeat 0 of {args.compare} vs repeat 0 of this run", per_seed, tol)
            print("(the hold target is each run's own reset pose: a difference in `reset q` also shifts the end values)")
        rows = [(name, by_name[name][1], 0, compare_traj(old_replay[name][0], replay[name][0])) for name in replay if name in old_replay]
        if rows:
            report_replay(f"REPLAY test across server runs: repeat 0 of {args.compare} vs repeat 0 of this run", rows)


if __name__ == "__main__":
    main()
