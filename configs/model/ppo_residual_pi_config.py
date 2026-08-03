"""Config for PPOResidualLearner: on-policy PPO trained on a residual policy
conditioned on the frozen VLA's own sampled base action.

Unlike PPOLearner, the actor here IS connected to the SFT-competent VLA: the
frozen VLA supplies the base action (sampled, never updated), and PPO trains
only a small residual correction on top of it, using the same TanhNormal +
PixelEditMultiplexer residual-actor construction ExpoFT itself uses. No
actor_pretrain_steps / pretrain_actor_bc here at all — there is no
"random-init actor with zero connection to the SFT policy" problem to work
around in the first place, since the base action already comes directly
from the SFT-competent frozen VLA. See ppo_residual.py's module docstring
for the full rationale.

Extends expo_ft_pi_config.py purely to reuse its shared infra fields
(pi05_config_name, encoder settings, etc.); the ExpoFT/SAC-specific fields
it inherits (N, n_edit_samples, fixed_temperature, critic_weight_decay,
num_qs, tau, ...) are unused by PPOResidualLearner and are never touched by
train_pi_robo.py's PPO override block.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "PPOResidualLearner"

    # PPO-specific hyperparameters. Overridable per-task via ppo_* fields in
    # the task YAML (train_pi_robo.py's PPOLearner override block is reused
    # as-is for this model_cls too -- same field names). Defaults below
    # match PPOResidualLearner.create()'s own Python-level defaults.
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.value_clip_eps = 0.2
    config.value_loss_coef = 0.5
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.num_minibatches = 4

    # Number of consecutive transitions per on-policy rollout -- same
    # meaning and same constraint (divisible by num_minibatches) as
    # PPOLearner's own. See BatchProcessor's on_policy mode.
    config.rollout_length = 512

    # Floor on the RESIDUAL's log-std (not the base VLA's -- the VLA is
    # frozen and has no log-std to speak of). Same value as PPOLearner's,
    # kept tight for the same numerical-stability reasons (see the
    # TanhNormal construction comment in ppo.py) even though the specific
    # BC-collapse failure mode that motivated it doesn't apply here (no BC
    # warm-start on this actor at all).
    config.actor_log_std_min = -5.0

    # Residual scale: the base VLA action is combined as
    # base_action + edit_scale * residual. Matches ExpoFT's own default so
    # the two are directly comparable at the same nominal correction budget.
    config.edit_scale = 0.2

    return config
