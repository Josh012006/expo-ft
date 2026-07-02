"""
Rigorous end-to-end validation: "if the SFT model perfectly cloned this demo,
would eval report success?"

For each of the first N recorded episodes, this script:
  1. Extracts the demo's raw actions using the SAME (fixed) logic as
     convert_maniskill_to_droid.py / convert_maniskill_to_lerobot.py.
  2. Loads the REAL norm_stats via the REAL build_pi05_config path (same code
     eval_policy.py uses) and builds openpi's actual Normalize/Unnormalize
     transforms from it.
  3. Round-trips each action through normalize -> unnormalize, simulating a
     perfectly-trained model reproducing the training target exactly.
  4. Resets the REAL eval env wrapper (expo_ft.env.env_factory.make_env_wrapper)
     to the EXACT recorded initial state (via reset_kwargs from the source
     trajectory JSON — this is what makes the test meaningful; a naive
     env.reset() randomizes the initial state and gives a false negative).
  5. Steps the reconstructed actions through the env and checks ManiSkill's
     own success signal.

Usage:
    python scripts/validate_demos_full_pipeline.py \
        --config configs/task/maniskill/stack_cube.yaml \
        --n-episodes 10

    python scripts/validate_demos_full_pipeline.py \
        --config configs/task/maniskill/pick_cube.yaml \
        --n-episodes 10 --verbose
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import h5py

from expo_ft.utils.config_loader import load_task_config, get_sft_config_name
from expo_ft.env.env_factory import make_env_wrapper


def load_norm_stats(cfg):
    """Replicate exactly how eval_policy.py / build_pi05_config resolve norm_stats,
    so this test uses the identical AssetsConfig (DROID official stats baked into
    the openpi config) that eval/SFT actually use — not a hand-rolled guess."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model_config", str(REPO_ROOT / "configs" / "model" / "expo_ft_pi_config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_config = mod.get_config()

    sft_config_name = get_sft_config_name(cfg)
    model_config.pi05_config_name = sft_config_name
    model_config.skip_repack_transforms = cfg.skip_repack_transforms
    # Deliberately NOT setting pi05_assets_dir/pi05_asset_id — see eval_policy.py
    # for why: the config's own baked AssetsConfig (DROID official) must win.

    from expo_ft.utils.train_utils import build_pi05_config
    _, pi05_train_config, _, _ = build_pi05_config(model_config)
    data_config = pi05_train_config.data.create(
        pi05_train_config.assets_dirs, pi05_train_config.model
    )
    if data_config.norm_stats is None:
        raise RuntimeError(
            f"No norm_stats resolved for config '{sft_config_name}'. Check that "
            "its AssetsConfig / assets_dirs actually point to a valid norm_stats.json."
        )
    print(f"Loaded norm_stats for '{sft_config_name}': keys={list(data_config.norm_stats.keys())}")
    return data_config.norm_stats, data_config.use_quantile_norm, pi05_train_config.model.action_dim


def extract_actions(raw_actions, cfg):
    """Mirror the (fixed) extraction logic in convert_maniskill_to_droid.py /
    convert_maniskill_to_lerobot.py: actions are already normalized to [-1,1]
    by ManiSkill's action_conversion step during the control-mode conversion
    replay — no further rescaling. Keep this in sync with those two scripts."""
    action_dim = getattr(cfg, "output_action_dim", 7)
    arm_dim = action_dim - 1
    arm = raw_actions[:, :arm_dim].astype(np.float32)
    grip = raw_actions[:, arm_dim:arm_dim + 1].astype(np.float32)
    return np.concatenate([arm, grip], axis=-1)


def roundtrip(normalize, unnormalize, norm_stats, action, padded_action_dim):
    """normalize -> pad -> unnormalize -> unpad a single raw action, simulating a
    perfectly trained model reproducing the SFT training target exactly.

    Mirrors Pi05Agent's real asymmetric flow: Normalize runs on the RAW action dim
    (matches training-time target computation), the model always outputs a FIXED
    padded_action_dim-wide vector, Unnormalize runs on that full padded width, and
    only then is it sliced back down to the raw action dim (Pi05Agent._pad_actions /
    _unpad_actions). Padding dims are trained toward 0, so we pad with zeros here too.

    Returns (reconstructed_action, max_abs_error).
    """
    raw_dim = action.shape[-1]

    state_kwarg = {}
    if "state" in norm_stats:
        state_dim = norm_stats["state"].mean.shape[-1]
        state_kwarg["state"] = np.zeros((1, state_dim), dtype=np.float32)

    # Step 1: normalize at the raw (unpadded) dim.
    normed = normalize({"actions": action[np.newaxis], **state_kwarg})
    normed_actions = np.asarray(normed["actions"])[0]

    # Step 2: pad to the model's fixed action_dim with zeros.
    padded = np.zeros((padded_action_dim,), dtype=np.float32)
    padded[:raw_dim] = normed_actions

    # Step 3: unnormalize at the padded width (matches inference-time model output).
    recon = unnormalize({"actions": padded[np.newaxis], **state_kwarg})
    a_recon_padded = np.asarray(recon["actions"])[0]

    # Step 4: unpad back to the raw action dim.
    a_recon = a_recon_padded[:raw_dim]

    err = float(np.max(np.abs(a_recon - action)))
    return a_recon, err


def main(config_path, traj_path, n_episodes, verbose, max_episode_steps=None):
    from mani_skill.utils import io_utils
    from openpi.transforms import Normalize, Unnormalize

    cfg = load_task_config(config_path)
    cfg.normalize_action = True  # match eval_policy.py
    if max_episode_steps is not None:
        print(f"Overriding cfg.max_episode_steps: {cfg.max_episode_steps} -> {max_episode_steps}")
        cfg.max_episode_steps = max_episode_steps

    norm_stats, use_quantiles, padded_action_dim = load_norm_stats(cfg)
    normalize = Normalize(norm_stats, use_quantiles=use_quantiles)
    unnormalize = Unnormalize(norm_stats, use_quantiles=use_quantiles)

    json_path = str(traj_path).replace(".h5", ".json")
    print(f"Reading trajectory metadata from: {json_path}")
    json_data = io_utils.load_json(json_path)
    episodes = json_data["episodes"]
    n_episodes = min(n_episodes, len(episodes))
    print(f"{len(episodes)} episodes available, testing first {n_episodes}")

    with h5py.File(traj_path, "r") as f:
        first_ep_id = episodes[0]["episode_id"]
        example_actions = extract_actions(np.array(f[f"traj_{first_ep_id}"]["actions"]), cfg)
        example_action = example_actions[0][np.newaxis]

        env = make_env_wrapper(
            env_creation_request={
                "example_action": example_action,
                "env_usage": "eval",
                "video_dir": None,
            },
            cfg=cfg,
        )

        successes = []
        max_errs = []

        for i in range(n_episodes):
            ep = episodes[i]
            ep_id = ep["episode_id"]
            traj_key = f"traj_{ep_id}"
            raw_actions = np.array(f[traj_key]["actions"])
            actions = extract_actions(raw_actions, cfg)

            reset_kwargs = dict(ep.get("reset_kwargs", {}) or {})
            env.reset(**reset_kwargs)

            success = False
            ep_max_err = 0.0
            steps_taken = 0
            for t, a in enumerate(actions):
                a_recon, err = roundtrip(normalize, unnormalize, norm_stats, a, padded_action_dim)
                ep_max_err = max(ep_max_err, err)

                if verbose and t < 3:
                    print(f"    ep{i} t{t}: raw={a[:3]}  recon={a_recon[:3]}  err={err:.6f}")

                env.step(a_recon.tolist())
                done, success, _, _ = env.get_info_for_step()
                steps_taken = t + 1
                if done:
                    break

            successes.append(success)
            max_errs.append(ep_max_err)
            print(
                f"Episode {i} (traj_{ep_id}): success={success}, "
                f"steps={steps_taken}, max round-trip err={ep_max_err:.6f}"
            )

        env.close()

    success_rate = float(np.mean(successes))
    print(f"\n{'=' * 55}")
    print(f"Success rate: {success_rate:.1%} ({sum(successes)}/{n_episodes})")
    print(f"Max round-trip error across all episodes: {max(max_errs):.6f}")
    if max(max_errs) > 1e-2:
        print(
            "WARNING: round-trip error is non-trivial — norm_stats normalize/"
            "unnormalize is not cleanly invertible for these actions. Check "
            "quantile clipping (actions outside q01/q99) or dtype issues."
        )
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--traj-path",
        default=None,
        help="Defaults to demos/<env_id>/motionplanning/trajectory.rgb.<control_mode>.physx_cpu.h5",
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-episode-steps", type=int, default=None,
        help="Override cfg.max_episode_steps for this run (test without editing the YAML)",
    )
    args = parser.parse_args()

    cfg_peek = load_task_config(args.config)
    traj_path = args.traj_path or str(
        REPO_ROOT
        / "demos"
        / cfg_peek.env_id
        / "motionplanning"
        / f"trajectory.rgb.{cfg_peek.control_mode}.physx_cpu.h5"
    )

    main(args.config, traj_path, args.n_episodes, args.verbose, args.max_episode_steps)
