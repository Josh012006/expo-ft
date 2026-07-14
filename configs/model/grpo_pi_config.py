"""Config for GRPOLearner: group-relative on-policy finetuning.

IMPORTANT: GRPOLearner's actor is a SEPARATE, small TanhNormal network (+
its own batch_encoder) — not the Pi0.5/SFT VLA itself. Same situation as
PPOLearner (see ppo_pi_config.py's docstring for the full rationale): the
loaded VLA is used only for input preprocessing / output denormalization,
never to initialize this network's own weights. See actor_pretrain_steps
below.

No critic/value baseline at all — advantage is computed relative to a group
of sampled rollouts from the same state, so there's no discount factor,
critic_lr, or GAE to configure (unlike PPO/SAC/ExpoFT). Extends
expo_ft_pi_config.py purely to reuse its shared infra fields (pi05_config_name,
encoder settings, etc.); the ExpoFT/SAC-specific fields it inherits (edit_scale,
critic_lr, discount, tau, num_qs, ...) are unused by GRPOLearner and are never
touched by train_pi_robo.py's GRPO override block.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "GRPOLearner"

    # GRPO-specific hyperparameters. Overridable per-task via grpo_* fields in
    # the task YAML (see train_pi_robo.py's GRPOLearner override block).
    # Defaults below match GRPOLearner.create()'s own Python-level defaults,
    # so behavior is identical whether or not a task YAML sets them.
    config.group_size = 4
    config.clip_eps = 0.2
    config.kl_coef = 0.04
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.num_minibatches = 4

    # If >0, run this many behavior-cloning (imitation) steps on offline
    # demo data to warm-start the actor (+ its own batch_encoder) BEFORE any
    # GRPO training starts — see grpo.py's pretrain_actor_bc(). 0 = disabled
    # (default; matches the pre-existing from-scratch-random-init behavior).
    config.actor_pretrain_steps = 0

    return config
