"""Config for GRPOLearner: group-relative on-policy finetuning of the VLA.

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

    return config
