"""Config for SACLearner: standard from-pixels SAC finetuning of the VLA.

Unlike EXPOLearner, SAC directly fine-tunes a single policy network (no
frozen base + residual edit policy, no target-critic argmax over candidates)
— the classic REDQ-style critic ensemble + auto-tuned temperature setup.

Extends expo_ft_pi_config.py purely to reuse its shared infra fields
(pi05_config_name, encoder settings, etc.). Unlike ppo_pi_config.py/
grpo_pi_config.py, no new fields are needed here: every SAC hyperparameter
(actor_lr, critic_lr, temp_lr, hidden_dims, discount, tau, num_qs, num_min_qs,
critic_dropout_rate, critic_weight_decay, critic_layer_norm, target_entropy,
entropy_scale, init_temperature) is already present, inherited transitively
from sac_config.py / td_config.py. The ExpoFT-only fields it also inherits
(edit_scale, N, n_edit_samples, adjust_target_entropy, fixed_temperature,
critic_grad_clip_norm, freeze_critic_encoder) are unused by SACLearner and
are never touched by train_pi_robo.py's SAC override block.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "SACLearner"

    return config
