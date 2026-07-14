"""Config for PPOLearner: on-policy PPO finetuning.

IMPORTANT: PPOLearner's actor is a SEPARATE, small TanhNormal network (+ its
own batch_encoder) — not the Pi0.5/SFT VLA itself. The loaded VLA
(`--config.pi05_weight_loader_path=...`) is used only for input
preprocessing / output denormalization (see ppo.py's load_agent()
docstring), never to initialize this network's own weights. This is a
deliberate workaround for Pi0.5 being flow-matching-based (no tractable
action log-probability, which PPO's math needs) — but it means the actor
otherwise starts from pure random initialization with zero connection to
the SFT policy's competence. See actor_pretrain_steps below.

Extends expo_ft_pi_config.py purely to reuse its shared infra fields
(pi05_config_name, encoder settings, etc.); the ExpoFT-specific fields it
inherits (edit_scale, N, n_edit_samples, fixed_temperature,
critic_weight_decay, critic_grad_clip_norm, freeze_critic_encoder, num_qs,
tau, ...) are unused by PPOLearner and are never touched by
train_pi_robo.py's PPO override block.
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

    # If >0, run this many behavior-cloning (imitation) steps on offline
    # demo data to warm-start the actor (+ its own batch_encoder) BEFORE any
    # PPO training starts — see ppo.py's pretrain_actor_bc(). 0 = disabled
    # (default; matches the pre-existing from-scratch-random-init behavior).
    config.actor_pretrain_steps = 0

    return config
