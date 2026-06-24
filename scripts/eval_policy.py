"""
Evaluate π₀.₅ policy on a ManiSkill task.

Usage:
    # Baseline (no checkpoint — base π₀.₅)
    python scripts/eval_policy.py --config configs/task/stack_cube.yaml --n-episodes 50

    # After SFT
    python scripts/eval_policy.py --config configs/task/stack_cube.yaml \
        --checkpoint logs/stack_cube/<run>/sft/stack_cube_sft/checkpoints/<step> \
        --n-episodes 50
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from expo_ft.utils.config_loader import load_task_config
from expo_ft.env.maniskill_env import ManiSkillEnvWrapper


def evaluate(cfg, checkpoint_path, n_episodes, seed):
    import jax
    import numpy as np
    import openpi.training.sharding as openpi_sharding
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.agents.alg.expo_ft import load_agent

    # Load model config
    from absl import flags
    from ml_collections import config_flags
    import sys

    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # Load a few demo transitions to get example_action and init replay buffer
    dataset = process_droid_dataset(
        cfg.droid_format_dir,
        cfg,
        num_data=1,
    )
    example_action = dataset[0]['actions'][np.newaxis]

    # Create env
    env = ManiSkillEnvWrapper(
        env_creation_request={
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": None,
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

    # Point to our local norm stats
    model_config.pi05_assets_dir = str(REPO_ROOT / "assets" / "expo_pi05_droid_lora_finetune_sft_cartesian_state")
    model_config.pi05_asset_id = cfg.lerobot_repo_id
    model_config.skip_repack_transforms = cfg.skip_repack_transforms

    # Build π₀.₅
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        model_config, seed, mesh, data_sharding, replicated_sharding,
        resume=False,
        default_prompt=cfg.language_instruction,
    )

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
            "base_image": replay_buffer.dataset_dict['base_image'][0][np.newaxis],
            "left_wrist_image": replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
            "state": replay_buffer.dataset_dict['state'][0][np.newaxis],
            "actions": replay_buffer.dataset_dict['actions'][0][np.newaxis],
        })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

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

    # Evaluation loop
    successes = []
    episode_lengths = []
    rng = jax.random.PRNGKey(seed)

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        steps = 0
        success = False
        from collections import deque
        action_plan = deque()

        while not done and steps < cfg.max_steps_per_episode:
            if not action_plan:
                action_chunk, agent, _ = agent.sample_actions(obs)
                action_plan.extend(action_chunk[:cfg.replan_steps])

            action = action_plan.popleft()
            _, _ = env.step(action.tolist())
            done, success, _, _ = env.get_info_for_step()
            obs = env.get_observation()
            steps += 1

        successes.append(float(success))
        episode_lengths.append(steps)

    env.close()

    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)

    print(f"\n{'='*40}")
    print(f"Episodes:     {n_episodes}")
    print(f"Success rate: {success_rate:.1%} ({int(sum(successes))}/{n_episodes})")
    print(f"Avg length:   {avg_length:.1f} steps")
    print(f"{'='*40}")

    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True,       help="Path to task YAML config")
    parser.add_argument("--checkpoint",  default=None,        help="Path to checkpoint (None = base π₀.₅)")
    parser.add_argument("--n-episodes",  type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
