"""Config for GRPOResidualLearner: on-policy GRPO trained on a residual
policy conditioned on the frozen VLA's own sampled base action.

Same relationship to GRPOLearner as PPOResidualLearner has to PPOLearner --
see that config's docstring and grpo_residual.py's module docstring for the
full rationale. No actor_pretrain_steps / pretrain_actor_bc here either, for
the same reason: the base action already comes from the SFT-competent
frozen VLA, so there's no random-init-actor problem to warm-start away.

No critic/value baseline (GRPO has none) -- advantage is computed relative
to a group of sampled rollouts from the same state. KL penalty here is
against a frozen snapshot of the RESIDUAL actor specifically (not the VLA,
which needs no such penalty since it's never updated).

Extends expo_ft_pi_config.py purely to reuse its shared infra fields; the
ExpoFT/SAC-specific fields it inherits are unused here and never touched by
train_pi_robo.py's GRPO override block.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "GRPOResidualLearner"

    # GRPO-specific hyperparameters. Overridable per-task via grpo_* fields
    # in the task YAML (train_pi_robo.py's GRPOLearner override block is
    # reused as-is for this model_cls too). Defaults below match
    # GRPOResidualLearner.create()'s own Python-level defaults.
    config.group_size = 4
    config.clip_eps = 0.2
    config.kl_coef = 0.04
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.num_minibatches = 4

    # Number of consecutive transitions per on-policy rollout -- same
    # meaning and constraint as GRPOLearner's own.
    config.rollout_length = 512

    # Floor on the RESIDUAL's log-std -- see ppo_residual_pi_config.py's
    # matching comment.
    config.actor_log_std_min = -5.0

    # Residual scale: base_action + edit_scale * residual. Matches ExpoFT's
    # own default (and PPOResidualLearner's) for direct comparability.
    config.edit_scale = 0.2

    return config
