"""
Evaluate π₀.₅ BASE model on a ManiSkill task.

Key differences from eval_policy.py:
  - No Unnormalize: π₀.₅ outputs actions in its internal normalized space [-1, 1]
  - normalize_action=True in ManiSkill: receives [-1, 1] and rescales to [-0.1, 0.1] m/step
  - No norm_stats needed: the pipeline is norm-free end-to-end
  - No SFT checkpoint: always uses base π₀.₅ weights

Usage:
    python scripts/eval_base_policy.py \
        --config configs/task/maniskill/stack_cube_eef.yaml \
        --n-episodes 50
"""

import argparse
import sys
import os
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from expo_ft.utils.config_loader import load_task_config, get_sft_config_name
from expo_ft.env.env_factory import make_env_wrapper


def evaluate(cfg, n_episodes, seed, video_dir=None):
    import jax
    import numpy as np
    import openpi.training.sharding as openpi_sharding
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.agents.alg.expo_ft import load_agent

    # Override normalize_action to True — π₀.₅ outputs in [-1,1],
    # ManiSkill rescales to physical units internally.
    cfg.normalize_action = True

    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # Load one demo to get example_action shape (not used for norm stats here)
    dataset = process_droid_dataset(
        cfg.droid_format_dir,
        cfg,
        num_data=1,
    )
    example_action = dataset[0]['actions'][np.newaxis]

    # Create env with normalize_action=True
    env = make_env_wrapper(
        env_creation_request={
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": video_dir,
        },
        cfg=cfg,
    )

    # Build model config
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model_config",
        str(REPO_ROOT / "configs/model/expo_ft_pi_config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_config = mod.get_config()

    sft_config_name = get_sft_config_name(cfg)
    model_config.pi05_config_name       = sft_config_name
    model_config.pi05_assets_dir        = str(REPO_ROOT / "assets" / sft_config_name)
    model_config.pi05_asset_id          = cfg.lerobot_repo_id
    model_config.skip_repack_transforms = cfg.skip_repack_transforms

    print("Using base π₀.₅ weights (no SFT checkpoint)")

    # Build π₀.₅
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        model_config, seed, mesh, data_sharding, replicated_sharding,
        resume=False,
        default_prompt=cfg.language_instruction,
    )

    # KEY: disable Unnormalize — π₀.₅ internal space [-1,1] goes directly to ManiSkill
    actor.output_transforms = actor._build_output_transform_pipeline(unnormalize=False)
    print("Unnormalize disabled — actions passed directly to ManiSkill in [-1, 1]")

    params = actor.get_params(actor_train_state)
    leaf = jax.tree_util.tree_leaves(params)[0]
    print(f"First param sum: {float(jax.numpy.sum(leaf)):.6f}")

    # Replay buffer init
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

    state_obs_key = cfg.state_obs_key
    action_dim    = getattr(cfg, 'output_action_dim', 7)

    successes      = []
    episode_lengths = []

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        obs  = env.reset()
        done = False
        steps = 0
        from collections import deque
        action_plan = deque()

        while not done and steps < cfg.max_steps_per_episode:
            if not action_plan:
                action_chunk, agent, _ = agent.sample_actions(obs, only_base_actions=True)
                if ep == 0 and steps == 0:
                    print(f"action_chunk shape: {action_chunk.shape}")
                    print(f"action_chunk[0]: {action_chunk[0]}")
                    print(f"action range: [{action_chunk.min():.3f}, {action_chunk.max():.3f}]")
                    print(f"obs {state_obs_key}: {obs[state_obs_key]}")
                action_plan.extend(action_chunk[:cfg.replan_steps])

            action = action_plan.popleft()
            _, _ = env.step(action.tolist())

            if ep == 0 and steps < 5:
                print(f"step {steps}: action={action[:action_dim]}, state={obs[state_obs_key][:3]}")

            done, success, _, _ = env.get_info_for_step()
            obs = env.get_observation()
            steps += 1

        successes.append(float(success))
        episode_lengths.append(steps)

    env.close()

    success_rate = np.mean(successes)
    print(f"\n{'='*40}")
    print(f"Episodes:     {n_episodes}")
    print(f"Success rate: {success_rate:.1%} ({int(sum(successes))}/{n_episodes})")
    print(f"Avg length:   {np.mean(episode_lengths):.1f} steps")
    print(f"{'='*40}")
    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = str(REPO_ROOT / "logs" / "eval_videos" / f"base_{timestamp}")

    evaluate(cfg=cfg, n_episodes=args.n_episodes, seed=args.seed, video_dir=video_dir)
