"""
Answers Jesse's exact question with certainty: is the reward entering the
learning agent sparse+binary 0/1, both in offline (demo) and online (rollout)
data? Two independent checks, since either one settles it on its own:

  1. Read env.reward_mode directly off the constructed env (same
     make_env_wrapper() construction path used everywhere else in this repo
     -- eval_policy.py, diagnose_reward_timing.py, etc.). gym.make() in
     maniskill_env.py never passes reward_mode explicitly, so whatever this
     prints IS ManiSkill's own default for this task -- not a guess.

  2. Independent of what the mode is CALLED, replay a real successful demo
     trajectory step-by-step through that same env and record the actual
     reward value at every step. Sparse binary reward looks like all-zero
     until exactly one terminal 1.0 (or -1/0/1 for fail/none/success);
     dense/normalized_dense reward is a shaped, usually-nonzero value at
     almost every step. This is the harder-to-fake check: it doesn't matter
     what the mode string claims if the actual numbers coming back tell a
     different story.

Usage:
    python scripts/check_reward_mode.py \
        --config configs/task/maniskill/push_cube_expo_ft.yaml \
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
    """Same extraction logic as convert_maniskill_to_droid.py / diagnose_reward_timing.py."""
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

        # Check 1: what does ManiSkill itself say its reward_mode is?
        print("=" * 70)
        try:
            raw_reward_mode = env._env.unwrapped.reward_mode
            print(f"env.unwrapped.reward_mode = {raw_reward_mode!r}")
        except AttributeError as e:
            print(f"Could not read reward_mode directly ({e}); relying on check 2 below.")
        print("=" * 70)

        all_rewards = []

        for ep_idx in range(n_episodes):
            ep = episodes[ep_idx]
            ep_id = ep["episode_id"]
            raw_actions = np.array(f[f"traj_{ep_id}"]["actions"])
            actions = extract_actions(raw_actions, cfg)

            reset_kwargs = dict(ep.get("reset_kwargs", {}) or {})
            env.reset(**reset_kwargs)

            ep_rewards = []
            for a in actions:
                env.step(a.tolist())
                _, _, reward, _ = env.get_info_for_step()
                ep_rewards.append(float(reward))

            all_rewards.extend(ep_rewards)
            n_zero = sum(1 for r in ep_rewards if abs(r) < 1e-6)
            n_total = len(ep_rewards)
            print(f"\nEpisode {ep_idx} (traj_{ep_id}), {n_total} steps:")
            print(f"  first 10 rewards: {[round(r, 3) for r in ep_rewards[:10]]}")
            print(f"  last 10 rewards:  {[round(r, 3) for r in ep_rewards[-10:]]}")
            print(f"  exactly-zero steps: {n_zero}/{n_total} ({100*n_zero/n_total:.1f}%)")
            print(f"  min={min(ep_rewards):.4f}  max={max(ep_rewards):.4f}  mean={np.mean(ep_rewards):.4f}")

        env.close()

        print(f"\n{'=' * 70}")
        print(f"ACROSS ALL {len(all_rewards)} STEPS:")
        distinct = sorted(set(round(r, 4) for r in all_rewards))
        print(f"  distinct reward values seen: {len(distinct)}"
              + (f"  ({distinct})" if len(distinct) <= 15 else f" (showing first/last 5): {distinct[:5]} ... {distinct[-5:]}"))
        n_zero_all = sum(1 for r in all_rewards if abs(r) < 1e-6)
        print(f"  exactly-zero fraction: {100*n_zero_all/len(all_rewards):.1f}%")
        if len(distinct) <= 3:
            print("  -> looks SPARSE (few distinct values) -- consistent with sparse/binary-ish reward.")
        else:
            print("  -> looks DENSE/SHAPED (many distinct values) -- NOT a sparse binary reward.")
        print(f"{'=' * 70}")


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
