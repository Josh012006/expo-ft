"""
Validates the RL-specific action-chunk construction that has NO analog in SFT:
PiReplayBuffer.insert()'s backfill mechanism, which turns single-step raw demo
actions into action_horizon-length chunks (the structure everything in RL --
critic training, argmax candidate scoring, misrank comparisons -- reads from).

validate_demos_full_pipeline.py already proves the SFT normalize/unnormalize
round-trip is correct for single-step actions replayed through the real env.
This script covers the piece that script does NOT touch: does chunk index k
of replay-buffer position i really hold action (i+k) from the ORIGINAL demo
trajectory, in the right order, with no off-by-one or reversal introduced by
the backfill loop in insert()?

Method, for the first N episodes:
  1. Load the same raw droid-format dataset process_droid_dataset() builds
     (single-step "actions" per transition, in original recorded order).
  2. Build a real replay buffer the SAME way create_replay_buffer() does for
     compare_argmax_vs_reference_q.py / normal RL training (same model_config,
     same norm_stats), and insert_dataset() the same episodes into it.
  3. For each buffer position i belonging to that episode (up to
     len(episode) - action_horizon), unnormalize dataset_dict["actions"][i]
     (shape [action_horizon, raw_action_dim]) back to physical units, and
     compare it -- position by position -- against the ORIGINAL raw actions
     at [i, i+1, ..., i+action_horizon-1] from the untouched demo trajectory.

Reports max error PER CHUNK POSITION (0..action_horizon-1) across all tested
transitions, not just a single aggregate max -- a reversal or off-by-one bug
would show up as one specific position having a much larger error than its
neighbors, not a uniform small numerical-precision error everywhere.

Usage:
    python scripts/validate_rl_action_chunks.py \
        --config configs/task/maniskill/push_cube_expo_ft.yaml \
        --n-episodes 5
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from expo_ft.utils.config_loader import load_task_config, get_sft_config_name
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.data.replay_buffer import create_replay_buffer


def build_model_config(cfg):
    """Same pattern as compare_argmax_vs_reference_q.py / eval_policy.py --
    build the REAL model config (ml_collections ConfigDict), not the task-YAML
    SimpleNamespace, since create_replay_buffer's `config` arg needs the
    former (see the 3 bugs fixed in compare_argmax_vs_reference_q.py)."""
    import importlib.util
    model_cls_name = getattr(cfg, "model_cls", "EXPOLearner")
    model_config_path = REPO_ROOT / (
        "configs/model/expo_ft_categorical_pi_config.py" if model_cls_name == "EXPOLearnerCategorical" else "configs/model/expo_ft_pi_config.py"
    )
    spec = importlib.util.spec_from_file_location("model_config", str(model_config_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model_config = mod.get_config()
    model_config.pi05_config_name = get_sft_config_name(cfg)
    model_config.skip_repack_transforms = cfg.skip_repack_transforms
    model_config.pi05_weight_loader_path = None
    return model_config


def main(config_path, n_episodes, seed):
    from openpi.transforms import Unnormalize
    from expo_ft.utils.train_utils import build_pi05_config

    cfg = load_task_config(config_path)
    cfg.normalize_action = True

    model_config = build_model_config(cfg)
    _, pi05_train_config, _, _ = build_pi05_config(model_config)
    data_config = pi05_train_config.data.create(pi05_train_config.assets_dirs, pi05_train_config.model)
    unnormalize = Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    action_horizon = pi05_train_config.model.action_horizon

    # Unnormalize needs a "state" entry present if norm_stats defines one --
    # it's not a no-op default, apply_tree raises if the key is simply absent.
    # Same pattern as validate_demos_full_pipeline.py's roundtrip(); sized to
    # action_horizon here since we unnormalize a whole chunk at once (each row
    # of the chunk is a separate action, so "state" needs a matching leading dim).
    state_kwarg = {}
    if "state" in data_config.norm_stats:
        state_dim = data_config.norm_stats["state"].mean.shape[-1]
        state_kwarg["state"] = np.zeros((action_horizon, state_dim), dtype=np.float32)

    print(f"Loading demo data from {cfg.droid_format_dir} ...")
    # Only load as many episodes as this run will actually check (+1 margin) --
    # loading all 550 episodes' images just to validate the first --n-episodes
    # is wasted memory and was OOM-killing this on a shared/limited node.
    dataset = process_droid_dataset(cfg.droid_format_dir, cfg, num_data=n_episodes + 1)
    example_action = dataset[0]["actions"][np.newaxis]

    # Recover per-episode boundaries from the raw dataset's own "dones" field,
    # in ORIGINAL recorded order -- this is our ground truth, untouched by
    # anything the replay buffer does.
    dones = np.array([bool(np.asarray(d["dones"]).item()) for d in dataset])
    ep_starts, ep_ends = [0], []
    for i, d in enumerate(dones):
        if d:
            ep_ends.append(i + 1)
            if i + 1 < len(dones):
                ep_starts.append(i + 1)
    if len(ep_ends) < len(ep_starts):
        ep_ends.append(len(dones))
    n_episodes = min(n_episodes, len(ep_starts))
    print(f"{len(ep_starts)} episodes available in the raw dataset, testing first {n_episodes}")

    raw_actions_all = np.stack([np.asarray(d["actions"]) for d in dataset])  # (T, raw_action_dim)

    replay_buffer = create_replay_buffer(
        config=model_config,
        example_action=example_action,
        capacity=len(dataset),
        task_description=cfg.language_instruction,
        replan_steps=cfg.replan_steps,
        seed=seed,
    )
    replay_buffer.insert_dataset(dataset)

    raw_action_dim = replay_buffer._raw_action_dim
    per_position_max_err = np.zeros(action_horizon)
    per_position_count = np.zeros(action_horizon, dtype=int)
    worst = {"err": -1.0, "ep": None, "i": None, "k": None}

    for ep_idx in range(n_episodes):
        start, end = ep_starts[ep_idx], ep_ends[ep_idx]
        ep_len = end - start
        last_valid_i = end - action_horizon  # need action_horizon future steps in-episode
        n_checked = max(0, last_valid_i - start)
        print(f"Episode {ep_idx}: buffer positions [{start}, {end}) (len={ep_len}), checking {n_checked} chunk positions")

        for i in range(start, last_valid_i):
            stored_chunk = np.asarray(replay_buffer.dataset_dict["actions"][i])  # (action_horizon, padded_action_dim), normalized
            recon = np.asarray(unnormalize({"actions": stored_chunk, **state_kwarg})["actions"])[:, :raw_action_dim]

            for k in range(action_horizon):
                original = raw_actions_all[i + k]
                err = float(np.max(np.abs(recon[k] - original)))
                per_position_max_err[k] = max(per_position_max_err[k], err)
                per_position_count[k] += 1
                if err > worst["err"]:
                    worst.update(err=err, ep=ep_idx, i=i, k=k)

    print(f"\n{'=' * 70}")
    print("Max round-trip error PER CHUNK POSITION (across all tested transitions):")
    for k in range(action_horizon):
        n = per_position_count[k]
        flag = "  <-- suspiciously high" if n > 0 and per_position_max_err[k] > 5e-2 else ""
        print(f"  position {k:2d} (should hold action t+{k}): max_err={per_position_max_err[k]:.6f}  (n={n}){flag}")

    print(f"\nWorst single case: episode {worst['ep']}, buffer index {worst['i']}, "
          f"chunk position {worst['k']}, err={worst['err']:.6f}")
    overall_max = float(np.max(per_position_max_err))
    print(f"Overall max round-trip error: {overall_max:.6f}")
    if overall_max > 5e-2:
        print(
            "\nWARNING: non-trivial round-trip error. If it's concentrated on ONE "
            "position (not spread evenly), suspect an off-by-one or reversal in "
            "PiReplayBuffer.insert()'s backfill loop. If it's spread evenly across "
            "all positions, suspect a normalize/unnormalize or quantile-clipping issue "
            "instead (same class of issue validate_demos_full_pipeline.py checks for)."
        )
    else:
        print(
            "\nOK: action chunks reconstruct the original per-step demo actions, in "
            "the right order, within numerical precision -- no evidence of an "
            "off-by-one, reversal, or corruption in the RL-specific chunking pipeline."
        )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config, args.n_episodes, args.seed)
