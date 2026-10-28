"""What of the previous episode is still there after the next reset?  (reset(seed) should not depend on what ran before)

Client, runs in .venv, AFTER forge_server.py is up. Do NOT run it while something else uses the server.

    python scripts/isaaclab/reset_history.py --config configs/task/isaaclab/peg_insert_forge_pi05.yaml \
        --demo /tmp/repro_demos/ep_0000_seed1.npz

Procedure, for every prefix length n of the demo (default 0 5 10 20 30 45 60 and the full demo):
    1. reset(demo seed), send the first n recorded actions open loop (n = 0: nothing)
    2. reset(probe seed) and write down the start state: joint angles, offset peg -> hole, finger opening
Before step 1 the simulation is always in the same situation (it has just been through a probe reset), so the ONLY thing
that changes from one n to the next is how far the arm went and what it touched. The start state after the probe reset is
compared with the one obtained for n = 0 (nothing moved). 0 = the reset forgot the previous episode.

Reading the table: the first n whose start state differs tells what leaks. A difference as soon as the arm has moved (small
n) = something about the arm / controller is not cleaned; only once the peg reaches the socket (large n) = the contacts of the
insertion leave a trace; never = the reset is clean for these episodes. Run it on a server with robot=franka as well: if the
Franka (no Robotiq, no taxel zones) is clean and the Robotiq is not, the leak comes from our Robotiq / tactile code, if both
leak it comes from FORGE / Isaac Sim itself.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.env_client import EnvClient  # noqa: E402


def snapshot(client, env_id, obs):
    """Start state: joint angles (rad), offset peg -> hole (mm), finger opening (mm)."""
    width = client._call_operation("debug", {"env_id": env_id, "kind": "finger_width"})["result"]
    return {"q": np.asarray(obs["observation/joint_position"], dtype=np.float64).reshape(-1),
            "off": 1000.0 * np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64).reshape(-1),
            "w": np.array([1000.0 * float(width)] if not isinstance(width, dict) else [np.nan])}


def diff(a, b):
    return {k: float(np.abs(a[k] - b[k]).max()) for k in ("q", "off", "w")}


def main():
    p = argparse.ArgumentParser(description="Does reset(seed) depend on the episode that ran before it?")
    p.add_argument("--config", required=True, help="Task YAML (host, port).")
    p.add_argument("--demo", required=True, help="A demo .npz (collect_waypoint_demos.py): its seed and actions are replayed.")
    p.add_argument("--prefix-steps", default="0,5,10,20,30,45,60,100000", help="Comma list of numbers of actions replayed before the probe reset.")
    p.add_argument("--probe-seeds", default="0,40", help="Comma list of seeds whose start state is measured after each prefix.")
    p.add_argument("--repeats", type=int, default=2, help="The whole list of prefixes is run this many times (odd repeats in reverse order).")
    args = p.parse_args()

    z = np.load(args.demo)
    demo_seed, actions = int(z["seed"]), np.asarray(z["actions"], dtype=np.float32)
    prefixes = sorted({min(int(v), len(actions)) for v in args.prefix_steps.split(",") if v.strip()})
    probes = [int(v) for v in args.probe_seeds.split(",") if v.strip()]
    if 0 not in prefixes:
        prefixes = [0] + prefixes
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    client = EnvClient(host=cfg.get("server_host", "localhost"), port=cfg.get("server_port", 8102))
    env_id, _ = client.create_env({})
    print(f"demo {Path(args.demo).name}: seed {demo_seed}, {len(actions)} actions; prefixes {prefixes}; probe seeds {probes}", flush=True)

    results = {ps: {n: [] for n in prefixes} for ps in probes}          # probe seed -> n -> [snapshot of repeat 0, 1, ...]
    for rep in range(args.repeats):
        for n in (prefixes if rep % 2 == 0 else prefixes[::-1]):
            for ps in probes:
                client.reset(env_id, demo_seed)
                for a in actions[:n]:
                    client.step(env_id, a)
                    client.get_observation(env_id)
                    done, _, _, _ = client.get_info_for_step(env_id)
                    if done:
                        break
                obs, _ = client.reset(env_id, ps)
                results[ps][n].append(snapshot(client, env_id, obs))
        print(f"repeat {rep} done", flush=True)

    for ps in probes:
        ref = results[ps][0][0]
        print(f"\nprobe seed {ps}: start state after reset({ps}) vs the one after a prefix of 0 actions (repeat 0)")
        print("prefix n | repeat | dq (rad) | doff (mm) | dw (mm)")
        first = None
        for n in prefixes:
            for rep, snap in enumerate(results[ps][n]):
                d = diff(snap, ref)
                changed = any(v != 0.0 for v in d.values())
                if changed and first is None:
                    first = n
                print(f"{n:8d} | {rep:6d} | {d['q']:8.3g} | {d['off']:9.3g} | {d['w']:7.3g}{'  <- differs' if changed else ''}")
        print(f"first prefix whose start state differs: {first if first is not None else 'none (the reset forgot the previous episode)'}")


if __name__ == "__main__":
    main()
