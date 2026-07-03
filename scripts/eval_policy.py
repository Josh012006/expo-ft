"""
Evaluate π₀.₅ policy on a ManiSkill task.

Usage:
    # After SFT (Recommended)
    python scripts/eval_policy.py --config configs/task/maniskill_stack_cube.yaml \
        --checkpoint logs/stack_cube/<run>/sft/<exp_name>/checkpoints/<step> \
        --n-episodes 50
"""

import argparse
import sys
import os
import datetime
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from expo_ft.utils.config_loader import load_task_config, get_sft_config_name
from expo_ft.env.env_factory import make_env_wrapper

def evaluate(cfg, checkpoint_path, n_episodes, seed, video_dir=None, collect_action_stats=False,
             episode_seeds=None, output_json=None, diagnose_subconditions=False):
    """
    episode_seeds: optional list[int] of length n_episodes. When provided, env.reset()
    is called with seed=episode_seeds[ep] for each episode — guarantees the SAME initial
    conditions (object/goal positions) across different checkpoints, for a fair
    apples-to-apples comparison. When None, falls back to unseeded random resets
    (existing behavior, unchanged).

    diagnose_subconditions: purely additive diagnostic, does NOT change success_rate
    or any reported metric. When True, at every step also reads env.get_raw_info()
    and tracks, per episode, whether each boolean sub-condition of a composite
    success criterion (e.g. PickCube-v1's is_obj_placed / is_robot_static) was ever
    True, and whether they were ever True AT THE SAME STEP. Printed as a summary at
    the end, alongside (not instead of) the normal success_rate report.
    """
    if episode_seeds is not None and len(episode_seeds) != n_episodes:
        raise ValueError(
            f"episode_seeds has {len(episode_seeds)} entries but n_episodes={n_episodes}"
        )
    import jax
    import numpy as np
    import openpi.training.sharding as openpi_sharding
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.env.droid_utils import make_dummy_transition
    from expo_ft.agents.alg.expo_ft import load_agent

    # CRITIQUE : ManiSkill DOIT clupper et rescale les deltas en [-0.1, 0.1]
    cfg.normalize_action = True

    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # Synthetic transition (shapes/dtypes only, from cfg) — no dataset on disk needed.
    # Eval never reads real demo content: env.reset()/get_observation() drive the rollout.
    dataset = make_dummy_transition(cfg)
    example_action = dataset[0]['actions'][np.newaxis]

    # Create env with explicit normalize_action=True
    env = make_env_wrapper(
        env_creation_request={
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": video_dir,
        },
        cfg=cfg,
    )

    # Load algorithm config
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model_config",
        str(REPO_ROOT / "configs/model/expo_ft_pi_config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_config = mod.get_config()

    # Override config dynamically from task YAML — no hardcoded values
    sft_config_name = get_sft_config_name(cfg)
    model_config.pi05_config_name  = sft_config_name
    model_config.skip_repack_transforms = cfg.skip_repack_transforms

    # IMPORTANT : ne PAS fixer pi05_assets_dir / pi05_asset_id ici.
    # La config openpi "expo_pi05_droid_lora_finetune_sft_joint_state" porte déjà
    # assets=AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
    #                      asset_id="droid")
    # bakée en dur dans training/config.py. Le SFT (run_pipeline.py stage_sft) n'override
    # jamais ce champ via CLI, donc l'entraînement lui-même a normalisé/dénormalisé avec
    # ces stats DROID officielles — pas avec compute_norm_stats.py sur nos démos locales.
    # En laissant ces deux champs vides ici, build_pi05_config() retombe sur l'AssetsConfig
    # bakée, ce qui garde l'eval cohérente avec l'entraînement, que checkpoint_path soit
    # fourni (SFT) ou non (baseline pi05_droid_jointpos pur).

    # Load SFT checkpoint if provided
    if checkpoint_path is not None:
        model_config.pi05_weight_loader_path = str(Path(checkpoint_path) / "params")
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        model_config.pi05_weight_loader_path = None
        print("Using base π₀.₅ weights (no checkpoint) — pi05_droid_jointpos + DROID norm_stats")

    # Build π₀.₅ (unnormalize est à True par défaut ici, ce qui est correct pour le SFT)
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        model_config, seed, mesh, data_sharding, replicated_sharding,
        resume=False,
        default_prompt=cfg.language_instruction,
    )

    # Print first param sum to verify checkpoint loading
    params = actor.get_params(actor_train_state)
    leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"First param sum: {float(jax.numpy.sum(leaf)):.6f}")

    # Create replay buffer for agent init
    replay_buffer = create_replay_buffer(
        config=model_config,
        example_action=example_action,
        capacity=100,
        task_description=cfg.language_instruction,
        replan_steps=cfg.replan_steps,
        seed=seed,
    )
    replay_buffer.insert_dataset(dataset)

    agent_example_observation, agent_example_state, agent_example_action = \
        replay_buffer.convert_to_critic_format({
            "base_image":       replay_buffer.dataset_dict['base_image'][0][np.newaxis],
            "left_wrist_image": replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
            "state":            replay_buffer.dataset_dict['state'][0][np.newaxis],
            "actions":          replay_buffer.dataset_dict['actions'][0][np.newaxis],
        })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim  = agent_example_state.squeeze().shape[-1]

    agent = load_agent(
        seed=seed,
        example_observation=agent_example_observation.squeeze(),
        example_action=agent_example_action.squeeze(),
        example_state=agent_example_state.squeeze(),
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        agent_kwargs=agent_kwargs,
        metadata=vla_metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=False,
        replan_steps=cfg.replan_steps,
        default_prompt=cfg.language_instruction,
        residual_action_xyzg=getattr(cfg, 'residual_action_xyzg', False),
    )
    agent = agent.cache_infer_params()

    # Config-driven keys
    state_obs_key  = cfg.state_obs_key
    action_dim     = getattr(cfg, 'output_action_dim', 7)

    # Evaluation loop
    successes = []
    subcond_summaries = []  # only populated if diagnose_subconditions=True
    all_actions = []  # collect raw actions for distribution analysis
    episode_lengths = []

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        if episode_seeds is not None:
            obs = env.reset(seed=int(episode_seeds[ep]))
        else:
            obs = env.reset()
        done = False
        steps = 0
        success = False
        ep_subcond = {}  # key -> bool, "ever True this episode" (diagnostic only)
        ep_subcond_together = False  # is_obj_placed AND is_robot_static ever True at the SAME step
        from collections import deque
        action_plan = deque()

        while not done and steps < cfg.max_steps_per_episode:
            if not action_plan:
                action_chunk, agent, _ = agent.sample_actions(obs, only_base_actions=True)
                all_actions.append(action_chunk)
                if ep == 0 and steps == 0:
                    print(f"\n--- DEBUG EP 0 STEP 0 ---")
                    print(f"action_chunk shape: {action_chunk.shape}")
                    print(f"action_chunk[0]: {action_chunk[0]}")
                    print(f"Dénormalisé (OpenPI) range - min: {action_chunk.min():.3f}, max: {action_chunk.max():.3f}")
                    print(f"obs {state_obs_key}: {obs.get(state_obs_key, 'MISSING')[:3]}")
                action_plan.extend(action_chunk[:cfg.replan_steps])

            action = action_plan.popleft()
            
            # Conversion en liste et envoi à ManiSkill
            env.step(action.tolist())

            if ep == 0 and steps == 0:
                print(f"prompt: {obs.get('prompt', 'MISSING')}")
                print(f"obs keys: {list(obs.keys())}")
            if ep == 0 and steps < 5:
                print(f"step {steps}: actionToSend={action[:action_dim]}, state={obs.get(state_obs_key, np.zeros(3))[:3]}")

            done, success, _, _ = env.get_info_for_step()
            if diagnose_subconditions:
                raw = env.get_raw_info()
                step_vals = {}
                for k, v in raw.items():
                    try:
                        step_vals[k] = bool(v.item()) if hasattr(v, "item") else bool(v)
                    except (ValueError, TypeError):
                        continue  # not a scalar bool-like field
                    ep_subcond[k] = ep_subcond.get(k, False) or step_vals[k]
                if step_vals.get("is_obj_placed") and step_vals.get("is_robot_static"):
                    ep_subcond_together = True
            obs = env.get_observation()
            steps += 1

        successes.append(float(success))
        if diagnose_subconditions:
            subcond_summaries.append({"ever": dict(ep_subcond), "ever_together": ep_subcond_together})
        episode_lengths.append(steps)

    env.close()

    success_rate = np.mean(successes)
    avg_length   = np.mean(episode_lengths)

    print(f"\n{'='*40}")
    print(f"Episodes:     {n_episodes}")
    print(f"Success rate: {success_rate:.1%} ({int(sum(successes))}/{n_episodes})")
    print(f"Avg length:   {avg_length:.1f} steps")
    print(f"{'='*40}")

    if diagnose_subconditions and subcond_summaries:
        keys = sorted({k for s in subcond_summaries for k in s["ever"].keys()})
        print("\n--- SUB-CONDITION DIAGNOSTIC (does not affect success_rate above) ---")
        for k in keys:
            ever_rate = np.mean([s["ever"].get(k, False) for s in subcond_summaries])
            print(f"  {k:20s} ever True: {ever_rate:.1%} ({sum(s['ever'].get(k, False) for s in subcond_summaries)}/{n_episodes} episodes)")
        together_rate = np.mean([s["ever_together"] for s in subcond_summaries])
        print(f"  {'is_obj_placed & is_robot_static (SAME step)':45s}: {together_rate:.1%} "
              f"({sum(s['ever_together'] for s in subcond_summaries)}/{n_episodes} episodes)")
        print("If 'ever True' is high for each condition individually but the 'SAME step' "
              "rate is much lower, the arm is reaching the goal and later settling, just "
              "not simultaneously — a stability/timing gap, not a failure to complete the task.")

    # Correction du crash de fin de fichier
    if all_actions and collect_action_stats:
        all_actions_np = np.concatenate(all_actions, axis=0)
        stats = {
            'mean': all_actions_np.mean(axis=0).tolist(),
            'std':  all_actions_np.std(axis=0).tolist(),
            'q01':  np.quantile(all_actions_np, 0.01, axis=0).tolist(),
            'q99':  np.quantile(all_actions_np, 0.99, axis=0).tolist(),
            'min':  all_actions_np.min(axis=0).tolist(),
            'max':  all_actions_np.max(axis=0).tolist(),
            'n_samples': len(all_actions_np)
        }
        print("\n--- COLLECTED ACTION STATS FROM INFERENCE ---")
        print(json.dumps(stats, indent=2))

    if output_json is not None:
        result = {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "n_episodes": n_episodes,
            "success_rate": success_rate,
            "avg_length": avg_length,
            "successes": successes,
            "episode_lengths": episode_lengths,
            "used_fixed_episode_seeds": episode_seeds is not None,
        }
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote results to: {output_json}")

    return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", default=None, help="Path to SFT directory checkpoint")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument(
        "--episode-seeds", default=None,
        help="Path to a JSON file containing a list of N per-episode seeds. When given, "
             "overrides --n-episodes to match the list length, and every episode is reset "
             "with its fixed seed instead of a random one — use this to compare multiple "
             "checkpoints on the EXACT same episodes (see scripts/eval_curve.py).",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Path to write a structured JSON result (success_rate, per-episode outcomes, "
             "etc). Used by scripts/eval_curve.py to aggregate across checkpoints.",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip saving rollout videos. Useful for large checkpoint sweeps where "
             "hundreds of videos would otherwise be written.",
    )
    parser.add_argument(
        "--diagnose-subconditions", action="store_true",
        help="Purely additive diagnostic — does not change success_rate. At every "
             "step, also reads the env's raw info dict and reports, per episode, "
             "whether each sub-condition of a composite success criterion (e.g. "
             "PickCube-v1's is_obj_placed / is_robot_static) was ever True, and "
             "whether they were ever True at the SAME step.",
    )
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    episode_seeds = None
    n_episodes = args.n_episodes
    if args.episode_seeds is not None:
        with open(args.episode_seeds) as f:
            episode_seeds = json.load(f)
        n_episodes = len(episode_seeds)
        print(f"Loaded {n_episodes} fixed episode seeds from: {args.episode_seeds}")

    mode = "sft" if args.checkpoint is not None else "baseline"
    if args.no_video:
        video_dir = None
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_dir = str(REPO_ROOT / "logs" / "eval_videos" / f"eval_{cfg.env_id}_{mode}_{timestamp}")

    evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        n_episodes=n_episodes,
        seed=args.seed,
        video_dir=video_dir,
        collect_action_stats=True,
        episode_seeds=episode_seeds,
        output_json=args.output_json,
        diagnose_subconditions=args.diagnose_subconditions,
    )
