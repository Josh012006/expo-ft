"""
Evaluate an RLinf π₀.₅ PyTorch checkpoint on a ManiSkill task.
Uses the model.safetensors directly — no JAX conversion needed.

Usage:
    python scripts/eval_policy_pytorch.py \
        --checkpoint assets/RLinf-Pi05-ManiSkill-25Main-SFT \
        --config configs/task/maniskill/stack_cube_rlinf.yaml \
        --n-episodes 50
"""

import argparse
import datetime
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from expo_ft.utils.config_loader import load_task_config
from expo_ft.env.env_factory import make_env_wrapper


def build_pytorch_policy(checkpoint_dir: str, task_description: str):
    """
    Load a RLinf π₀.₅ PyTorch checkpoint and create an openpi Policy.
    Directly mirrors RLinf's toolkits/eval_scripts_openpi/__init__.py.
    """
    import json
    import safetensors.torch as st
    import openpi.models.pi0_config as pi0_config
    import openpi.transforms as transforms
    import openpi.policies.policy as _policy
    from openpi.models_pytorch import pi0_pytorch
    from openpi.policies import droid_policy

    checkpoint_dir = Path(checkpoint_dir)
    weight_path = checkpoint_dir / "model.safetensors"
    assert weight_path.exists(), f"model.safetensors not found in {checkpoint_dir}"

    # Load norm stats
    norm_stats_path = checkpoint_dir / "physical-intelligence" / "maniskill" / "norm_stats.json"
    with open(norm_stats_path) as f:
        raw = json.load(f)["norm_stats"]

    norm_stats = {
        key: transforms.NormStats(
            mean=np.array(stats["mean"], dtype=np.float32),
            std=np.array(stats["std"], dtype=np.float32),
        )
        for key, stats in raw.items()
    }

    # Build model — config from metadata.pt
    import torch
    meta = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    model_cfg_dict = meta["config"]["model"]
    config = pi0_config.Pi0Config(
        pi05=model_cfg_dict.get("pi05", True),
        action_dim=model_cfg_dict.get("action_dim", 32),
        action_horizon=model_cfg_dict.get("action_horizon", 5),
        paligemma_variant=model_cfg_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_cfg_dict.get("action_expert_variant", "gemma_300m"),
    )
    model = pi0_pytorch.PI0Pytorch(config)
    st.load_model(model, str(weight_path), strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    print(f"Model loaded from {weight_path}")

    droid_inputs = droid_policy.DroidInputs(model_type=config.model_type)

    policy = _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(task_description),
            droid_inputs,
            transforms.Normalize(norm_stats),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats),
        ],
        sample_kwargs={"num_steps": meta["config"]["model"].get("num_steps", 4)},
        is_pytorch=True,
        pytorch_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return policy


def evaluate(cfg, checkpoint_dir, n_episodes, seed, video_dir=None):
    # Build env
    env = make_env_wrapper(
        env_creation_request={
            "env_usage": "eval",
            "video_dir": video_dir,
        },
        cfg=cfg,
    )

    # Build policy
    policy = build_pytorch_policy(checkpoint_dir, cfg.language_instruction)

    # Eval loop
    successes = []
    episode_lengths = []

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < cfg.max_episode_steps:
            # Policy expects obs dict with openpi keys
            action_chunk = policy.infer(obs)["actions"]  # (action_horizon, action_dim)

            for action in action_chunk[:cfg.replan_steps]:
                _, _ = env.step(action.tolist())
                done, success, _, _ = env.get_info_for_step()
                obs = env.get_observation()
                steps += 1
                if done:
                    break

        _, success, _, _ = env.get_info_for_step()
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
    parser.add_argument("--checkpoint", required=True, help="Path to RLinf checkpoint dir (contains model.safetensors)")
    parser.add_argument("--config",     required=True, help="Path to task YAML config")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = str(REPO_ROOT / "logs" / "eval_videos" / f"rlinf_pytorch_{timestamp}")

    evaluate(
        cfg=cfg,
        checkpoint_dir=args.checkpoint,
        n_episodes=args.n_episodes,
        seed=args.seed,
        video_dir=video_dir,
    )
