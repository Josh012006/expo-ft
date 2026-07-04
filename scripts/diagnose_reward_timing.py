"""
Non-invasive diagnostic: empirically observe the exact timing between actions
and their recorded reward/done in train_pi_robo.py's transition construction,
WITHOUT modifying train_pi_robo.py itself.

Context: train_pi_robo.py reads env.get_info_for_step() at the TOP of each loop
iteration (before calling env.step() for that iteration's action), then later
builds transition_dict = dict(observations=<this info>, actions=<this action>,
rewards=<this info's reward>, ...). This script replays a KNOWN successful demo
through the same env wrapper, in the same read-before/step/read-after order,
and prints — for every step — what the "top of iteration" info says vs what a
FRESH call to get_info_for_step() says immediately after actually taking that
step. This shows plainly which iteration index the true success/reward lands
on, versus which iteration index train_pi_robo.py would attribute it to.

We are NOT proposing to change train_pi_robo.py based on this: the exact same
read-before-step pattern exists verbatim in the original ExpoFT reference
implementation (github.com/pd-perry/expo-ft, train_pi_robo.py lines ~254-273),
so this is confirmed to be the original, validated design — not a bug
introduced when porting to ManiSkill. This script is for understanding /
documentation only.

Usage:
    python scripts/diagnose_reward_timing.py \
        --config configs/task/maniskill/stack_cube.yaml \
        --n-episodes 2
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import h5py

from expo_ft.utils.config_loader import load_task_config
from expo_ft.env.env_factory import make_env_wrapper


def extract_actions(raw_actions, cfg):
    """Same extraction logic as convert_maniskill_to_droid.py (post-fix):
    actions are already normalized [-1,1], no rescaling."""
    action_dim = getattr(cfg, "output_action_dim", 7)
    arm_dim = action_dim - 1
    arm = raw_actions[:, :arm_dim].astype(np.float32)
    grip = raw_actions[:, arm_dim:arm_dim + 1].astype(np.float32)
    return np.concatenate([arm, grip], axis=-1)


def main(config_path, traj_path, n_episodes):
    from mani_skill.utils import io_utils

    cfg = load_task_config(config_path)
    cfg.normalize_action = True

    json_path = str(traj_path).replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    episodes = json_data["episodes"]
    n_episodes = min(n_episodes, len(episodes))

    with h5py.File(traj_path, "r") as f:
        first_ep_id = episodes[0]["episode_id"]
        example_actions = extract_actions(np.array(f[f"traj_{first_ep_id}"]["actions"]), cfg)
        example_action = example_actions[0][np.newaxis]

        env = make_env_wrapper(
            env_creation_request={"example_action": example_action, "env_usage": "train", "video_dir": None},
            cfg=cfg,
        )

        for ep_idx in range(n_episodes):
            ep = episodes[ep_idx]
            ep_id = ep["episode_id"]
            raw_actions = np.array(f[f"traj_{ep_id}"]["actions"])
            actions = extract_actions(raw_actions, cfg)

            reset_kwargs = dict(ep.get("reset_kwargs", {}) or {})
            env.reset(**reset_kwargs)

            print(f"\n=== Episode {ep_idx} (traj_{ep_id}), {len(actions)} steps ===")
            print(f"{'step':>4} | {'top-of-iter done/reward':^28} | {'true post-step done/reward':^28} | attribution")
            print("-" * 95)

            true_success_step = None
            for t, a in enumerate(actions):
                # Mimic train_pi_robo.py: read info BEFORE stepping (what iteration t
                # would read as "this iteration's done/reward" at the top of the loop).
                done_before, success_before, reward_before, mask_before = env.get_info_for_step()

                env.step(a.tolist())

                # Now read the TRUE immediate outcome of the action just taken.
                done_true, success_true, reward_true, mask_true = env.get_info_for_step()

                if success_true and true_success_step is None:
                    true_success_step = t

                note = ""
                if t == true_success_step:
                    note = "<-- true success happens HERE"
                if true_success_step is not None and t == true_success_step + 1:
                    note = "<-- train_pi_robo.py's transition_dict at THIS iteration would record reward/done=True (one step later)"

                print(f"{t:>4} | done={done_before!s:<5} r={reward_before:<6.2f}         "
                      f"| done={done_true!s:<5} r={reward_true:<6.2f}          | {note}")

            if true_success_step is None:
                print("(no success reached in this episode)")

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--traj-path", default=None)
    parser.add_argument("--n-episodes", type=int, default=2)
    args = parser.parse_args()

    cfg_peek = load_task_config(args.config)
    traj_path = args.traj_path or str(
        REPO_ROOT / "demos" / cfg_peek.env_id / "motionplanning"
        / f"trajectory.rgb.{cfg_peek.control_mode}.physx_cpu.h5"
    )
    main(args.config, traj_path, args.n_episodes)
