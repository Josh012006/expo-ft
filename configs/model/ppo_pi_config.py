"""Config for PPOLearner: on-policy PPO finetuning of the VLA.

Unlike EXPOLearner, PPO fully fine-tunes the VLA on-policy via GAE/clipped
surrogate loss — no frozen base + residual edit policy, no target-critic
argmax over candidates. Extends expo_ft_pi_config.py purely to reuse its
shared infra fields (pi05_config_name, encoder settings, etc.); the
ExpoFT-specific fields it inherits (edit_scale, N, n_edit_samples,
fixed_temperature, critic_weight_decay, critic_grad_clip_norm,
freeze_critic_encoder, num_qs, tau, ...) are unused by PPOLearner and are
never touched by train_pi_robo.py's PPO override block.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "PPOLearner"

    # PPO-specific hyperparameters. Overridable per-task via ppo_* fields in
    # the task YAML (see train_pi_robo.py's PPOLearner override block).
    # Defaults below match PPOLearner.create()'s own Python-level defaults,
    # so behavior is identical whether or not a task YAML sets them.
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.value_clip_eps = 0.2
    config.value_loss_coef = 0.5
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.num_minibatches = 4

    return config
