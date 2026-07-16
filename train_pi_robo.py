#! /usr/bin/env python
import os
import logging
import time
from collections import deque

import numpy as np
import tqdm
from absl import app, flags

from ml_collections import config_flags

import jax
import etils.epath as epath

import wandb
from expo_ft.agents import initialize_checkpoint_dir, save_replay_buffer_transition
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.data.batch_processor import BatchProcessor
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.env.droid_utils import process_droid_dataset
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb
from expo_ft.utils.config_loader import load_task_config, resolve_run_dir


import openpi.training.sharding as openpi_sharding

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from torch.utils.tensorboard import SummaryWriter


FLAGS = flags.FLAGS

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices for sharding.")
flags.DEFINE_string("task_config", "configs/task/stack_cube.yaml", "Path to task YAML config.")


config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def main(_):
    init_logging()

    # Load task config from YAML
    cfg = load_task_config(FLAGS.task_config)
    # Override pi05_config_name dynamically from task config
    from expo_ft.utils.config_loader import get_sft_config_name
    FLAGS.config.pi05_config_name = get_sft_config_name(cfg)
    FLAGS.config.skip_repack_transforms = cfg.skip_repack_transforms

    # Read once, early — used both for the hyperparameter overrides below and
    # for the learner-import dispatch further down. Reading it here (rather
    # than only later, where it used to be read) does not change any
    # behavior — FLAGS.config.model_cls is already fully populated at this
    # point since ml_collections loads --config before main() runs.
    model_cls = FLAGS.config.model_cls

    # Override FLAGS.config RL hyperparameters from the task YAML so everything
    # is configured in one place (the YAML) rather than split between YAML and
    # configs/model/*.py. Each learner reads its own, differently-prefixed
    # fields (rl_* for EXPOLearner, ppo_* for PPOLearner, grpo_* for
    # GRPOLearner) so a single task YAML can hold tuned overrides for every
    # algorithm at once without collisions.
    # NOTE: float() wrapping below is a deliberate defense against a PyYAML quirk —
    # bare scientific notation without a decimal point (e.g. "3e-4") is parsed as
    # a STRING, not a float (needs "3.0e-4" to parse correctly). ml_collections
    # then raises a TypeError trying to assign a str into a float-typed field.
    # float(x) is a no-op if x is already a float, and fixes it if x is a
    # not-quite-valid-YAML-float string — belt and suspenders alongside fixing
    # the YAML values themselves.
    if model_cls in ("EXPOLearner", "EXPOLearnerOld"):
        # --- unchanged from before this refactor: byte-for-byte identical ---
        FLAGS.config.actor_lr         = float(getattr(cfg, "rl_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr        = float(getattr(cfg, "rl_lr", FLAGS.config.critic_lr))
        FLAGS.config.discount         = float(getattr(cfg, "rl_discount", FLAGS.config.discount))
        FLAGS.config.tau              = float(getattr(cfg, "rl_tau", FLAGS.config.tau))
        FLAGS.config.init_temperature = float(getattr(cfg, "rl_init_temperature", FLAGS.config.init_temperature))
        FLAGS.config.adjust_target_entropy = getattr(cfg, "rl_adjust_target_entropy", FLAGS.config.adjust_target_entropy)
        _rl_fixed_temperature = getattr(cfg, "rl_fixed_temperature", FLAGS.config.fixed_temperature)
        FLAGS.config.fixed_temperature = float(_rl_fixed_temperature) if _rl_fixed_temperature is not None else None
        _rl_critic_weight_decay = getattr(cfg, "rl_critic_weight_decay", FLAGS.config.critic_weight_decay)
        FLAGS.config.critic_weight_decay = float(_rl_critic_weight_decay) if _rl_critic_weight_decay is not None else None
        _rl_critic_grad_clip_norm = getattr(cfg, "rl_critic_grad_clip_norm", FLAGS.config.critic_grad_clip_norm)
        FLAGS.config.critic_grad_clip_norm = float(_rl_critic_grad_clip_norm) if _rl_critic_grad_clip_norm is not None else None
        FLAGS.config.freeze_critic_encoder = getattr(cfg, "rl_freeze_critic_encoder", FLAGS.config.freeze_critic_encoder)
        if hasattr(cfg, "rl_hidden_dims"):
            FLAGS.config.hidden_dims  = tuple(cfg.rl_hidden_dims)
        FLAGS.config.edit_scale       = float(getattr(cfg, "rl_edit_scale", FLAGS.config.edit_scale))
        # --- end of original ExpoFT block; critic_pretrain_steps added below is
        # a new, default-off (0) field — behavior for existing configs that
        # don't set rl_critic_pretrain_steps is unchanged ---
        FLAGS.config.critic_pretrain_steps = int(getattr(cfg, "rl_critic_pretrain_steps", FLAGS.config.critic_pretrain_steps))
        FLAGS.config.num_atoms = int(getattr(cfg, "rl_num_atoms", FLAGS.config.num_atoms))
        FLAGS.config.v_min = float(getattr(cfg, "rl_v_min", FLAGS.config.v_min))
        FLAGS.config.v_max = float(getattr(cfg, "rl_v_max", FLAGS.config.v_max))
        FLAGS.config.reward_scale_decay = float(getattr(cfg, "rl_reward_scale_decay", FLAGS.config.reward_scale_decay))
        FLAGS.config.kl_coef = float(getattr(cfg, "rl_kl_coef", FLAGS.config.kl_coef))
        FLAGS.config.entropy_scale = float(getattr(cfg, "rl_entropy_scale", FLAGS.config.entropy_scale))
        FLAGS.config.kl_ref_std = float(getattr(cfg, "rl_kl_ref_std", FLAGS.config.kl_ref_std))
    elif model_cls == "PPOLearner":
        FLAGS.config.actor_lr  = float(getattr(cfg, "ppo_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr = float(getattr(cfg, "ppo_lr", FLAGS.config.critic_lr))
        FLAGS.config.discount  = float(getattr(cfg, "ppo_discount", FLAGS.config.discount))
        FLAGS.config.gae_lambda        = float(getattr(cfg, "ppo_gae_lambda", FLAGS.config.gae_lambda))
        FLAGS.config.clip_eps          = float(getattr(cfg, "ppo_clip_eps", FLAGS.config.clip_eps))
        FLAGS.config.value_loss_coef   = float(getattr(cfg, "ppo_value_loss_coef", FLAGS.config.value_loss_coef))
        FLAGS.config.entropy_coef      = float(getattr(cfg, "ppo_entropy_coef", FLAGS.config.entropy_coef))
        _ppo_value_clip_eps = getattr(cfg, "ppo_value_clip_eps", FLAGS.config.value_clip_eps)
        FLAGS.config.value_clip_eps = float(_ppo_value_clip_eps) if _ppo_value_clip_eps is not None else None
        _ppo_max_grad_norm = getattr(cfg, "ppo_max_grad_norm", FLAGS.config.max_grad_norm)
        FLAGS.config.max_grad_norm = float(_ppo_max_grad_norm) if _ppo_max_grad_norm is not None else None
        FLAGS.config.num_minibatches = int(getattr(cfg, "ppo_num_minibatches", FLAGS.config.num_minibatches))
        if hasattr(cfg, "ppo_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.ppo_hidden_dims)
        FLAGS.config.actor_pretrain_steps = int(getattr(cfg, "ppo_actor_pretrain_steps", FLAGS.config.actor_pretrain_steps))
    elif model_cls == "GRPOLearner":
        FLAGS.config.actor_lr     = float(getattr(cfg, "grpo_lr", FLAGS.config.actor_lr))
        FLAGS.config.group_size   = int(getattr(cfg, "grpo_group_size", FLAGS.config.group_size))
        FLAGS.config.clip_eps     = float(getattr(cfg, "grpo_clip_eps", FLAGS.config.clip_eps))
        FLAGS.config.kl_coef      = float(getattr(cfg, "grpo_kl_coef", FLAGS.config.kl_coef))
        FLAGS.config.entropy_coef = float(getattr(cfg, "grpo_entropy_coef", FLAGS.config.entropy_coef))
        _grpo_max_grad_norm = getattr(cfg, "grpo_max_grad_norm", FLAGS.config.max_grad_norm)
        FLAGS.config.max_grad_norm = float(_grpo_max_grad_norm) if _grpo_max_grad_norm is not None else None
        FLAGS.config.num_minibatches = int(getattr(cfg, "grpo_num_minibatches", FLAGS.config.num_minibatches))
        if hasattr(cfg, "grpo_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.grpo_hidden_dims)
        FLAGS.config.actor_pretrain_steps = int(getattr(cfg, "grpo_actor_pretrain_steps", FLAGS.config.actor_pretrain_steps))
    elif model_cls == "SACLearner":
        FLAGS.config.actor_lr  = float(getattr(cfg, "sac_lr", FLAGS.config.actor_lr))
        FLAGS.config.critic_lr = float(getattr(cfg, "sac_lr", FLAGS.config.critic_lr))
        FLAGS.config.discount  = float(getattr(cfg, "sac_discount", FLAGS.config.discount))
        FLAGS.config.tau       = float(getattr(cfg, "sac_tau", FLAGS.config.tau))
        FLAGS.config.init_temperature = float(getattr(cfg, "sac_init_temperature", FLAGS.config.init_temperature))
        _sac_target_entropy = getattr(cfg, "sac_target_entropy", FLAGS.config.target_entropy)
        FLAGS.config.target_entropy = float(_sac_target_entropy) if _sac_target_entropy is not None else None
        _sac_critic_weight_decay = getattr(cfg, "sac_critic_weight_decay", FLAGS.config.critic_weight_decay)
        FLAGS.config.critic_weight_decay = float(_sac_critic_weight_decay) if _sac_critic_weight_decay is not None else None
        FLAGS.config.num_qs = int(getattr(cfg, "sac_num_qs", FLAGS.config.num_qs))
        if hasattr(cfg, "sac_hidden_dims"):
            FLAGS.config.hidden_dims = tuple(cfg.sac_hidden_dims)
    # BCLearner: no RL hyperparameter overrides needed — imitation-only,
    # no critic/GAE/advantage machinery to tune here.

    # Sync actor_success_only from the task YAML into the model config too —
    # BatchProcessor already reads it from cfg (line below, via train_pi_robo's
    # own actor_success_only variable), but the EXPOLearner agent itself reads
    # its own copy from FLAGS.config, which defaults to True in
    # expo_ft_pi_config.py. Without this sync, BatchProcessor (correctly seeing
    # the YAML's actor_success_only=False) never builds an actor_batch, while
    # the agent (still seeing FLAGS.config's default True) expects one — crash.
    FLAGS.config.actor_success_only = getattr(cfg, "actor_success_only", False)
    # AssetsConfig DROID officielle deja bakee dans la config openpi nommee ci-dessus —
    # ne pas l'ecraser ici (meme bug corrige dans eval_policy.py : le SFT a ete
    # entraine avec ces stats officielles, pas des stats locales par repo_id).
    # Mark the run directory as an RL run (e.g. stack_cube_expo_ft_2026-07-05_01-06-12_rl)
    # so it's visually distinguishable from an SFT run directory at a glance.
    run_dir, resuming = resolve_run_dir(cfg, resume_dir=cfg.rl_resume_dir, suffix="rl")

    # getattr with a default here since offline_ratio no longer exists at all
    # in the PPO/GRPO task YAMLs (removed — it was dead for them regardless,
    # see is_on_policy_algo below) whereas EXPOLearner/SACLearner/BCLearner
    # YAMLs still define it and this assert still validates their real value.
    assert 0.0 <= getattr(cfg, "offline_ratio", 0.0) <= 1.0

    if cfg.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {cfg.batch_size} must be divisible by "
            f"the number of devices {jax.device_count()}"
        )
    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    log_dir = run_dir
    train_video_dir = os.path.join(log_dir, "train_videos")
    os.makedirs(train_video_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard writer — logs saved alongside checkpoints
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    checkpoint_dir_path = epath.Path(checkpoint_dir)
    checkpoint_manager, resuming = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=cfg.keep_period,
        max_to_keep=getattr(cfg, "max_to_keep", 100),
        # A fresh run gets its own brand-new timestamped directory (see
        # resolve_run_dir) — safe to "overwrite" since it's empty. Only skip
        # this when we're genuinely resuming (cfg.rl_resume_dir set), so we never
        # risk wiping real checkpoints being resumed into.
        overwrite=not resuming,
        resume=resuming,
    )

    init_wandb(checkpoint_dir_path, resuming, cfg.project_name, cfg.run_name)
    wandb.config.update(vars(cfg), allow_val_change=resuming)

    if cfg.env_type in ('droid', 'sim'):
        dataset = process_droid_dataset(
            cfg.droid_format_dir,
            cfg,
            num_data=cfg.num_data_rl if cfg.num_data_rl > 0 else None,
        )
        example_action = dataset[0]['actions'][np.newaxis]
    else:
        raise ValueError(f"Unsupported env_type: {cfg.env_type}")

    # Load env wrapper dynamically from task config
    train_env_creation_request = {
        "example_action": example_action,
        "env_usage": "train",
        "video_dir": train_video_dir,
    }

    logging.info("Creating environment...")
    if cfg.env_wrapper == "droid":
        from expo_ft.env.env_client import EnvClientWrapper
        env = EnvClientWrapper(
            env_creation_request=train_env_creation_request,
            host="localhost",
            port=8102,
        )
    else:
        from expo_ft.env.env_factory import make_env_wrapper
        env = make_env_wrapper(env_creation_request=train_env_creation_request, cfg=cfg)
    env.reset()
    logging.info(f"Created training environment {env.env_id}")

    # model_cls already read near the top of main() (see hyperparameter
    # override block above) — reused here for the learner-import dispatch.
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearnerOld":
        # The original, reference-faithful architecture (MSE scalar critic,
        # REDQ-style ensemble) preserved unmodified in expo_ft_old.py — for
        # direct A/B comparison against the categorical (XQC/XQCfD-style)
        # rewrite now used by "EXPOLearner". Shares the exact same config
        # overrides above (expo_ft_old.create()'s signature ends in **kwargs,
        # so the categorical-specific fields set there — num_atoms, v_min,
        # kl_coef, etc. — are silently absorbed and ignored, not an error).
        from expo_ft.agents.alg.expo_ft_old import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "PPOLearner":
        from expo_ft.agents.alg.ppo import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "GRPOLearner":
        from expo_ft.agents.alg.grpo import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "SACLearner":
        from expo_ft.agents.alg.sac import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    # PPO/GRPO are genuinely on-policy — their update() math (importance ratio
    # against old_log_probs, GAE bootstrap, GRPO's group-relative advantage
    # over CONTIGUOUS same-group rollouts) assumes every transition in a batch
    # was actually sampled from a known, recent version of the CURRENT policy.
    # Demo transitions (scripted motion-planning actions) were never sampled
    # from any version of the trained policy, so mixing them in — whether via
    # offline_ratio>0 (a separate offline buffer blended into every batch) or
    # via offline_ratio==0 (which instead seeds them permanently into the
    # ONLINE replay buffer, where uniform sampling would keep resurfacing them
    # indefinitely) — silently breaks that assumption and any group structure.
    # Force zero demo contamination for these two model classes, regardless of
    # whatever offline_ratio happens to be set to in the task YAML.
    is_on_policy_algo = model_cls in ("PPOLearner", "GRPOLearner")
    if is_on_policy_algo and getattr(cfg, "offline_ratio", 0.0) != 0:
        logging.warning(
            "model_cls=%s is on-policy — ignoring offline_ratio=%s from the task "
            "YAML and forcing zero demo contamination (no dataset inserted into "
            "either replay buffer for actual training sampling).",
            model_cls, getattr(cfg, "offline_ratio", 0.0),
        )

    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        FLAGS.config, cfg.seed, mesh, data_sharding, replicated_sharding,
        resuming, env.task_description,
    )

    rb_args = dict(
        config=FLAGS.config,
        example_action=example_action,
        capacity=cfg.max_steps,
        task_description=env.task_description,
        replan_steps=cfg.replan_steps,
        seed=cfg.seed,
    )
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)

    actor_success_only = getattr(cfg, "actor_success_only", False)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=cfg.batch_size,
        utd_ratio=cfg.utd_ratio,
        offline_ratio=0.0 if is_on_policy_algo else cfg.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=None if is_on_policy_algo else dataset,
    )

    if is_on_policy_algo:
        # offline_replay_buffer still needs at least one transition inserted
        # so the shape-inference call just below (convert_to_critic_format on
        # offline_replay_buffer.dataset_dict[...]) has something to read.
        # This is ONLY for shape inference — BatchProcessor above got
        # dataset=None and offline_ratio=0.0, so it never samples from this
        # buffer for actual training batches; replay_buffer (the online one)
        # stays completely free of demo data too, filled only by genuine
        # on-policy rollout transitions collected from here on.
        offline_replay_buffer.insert_dataset(dataset)

    agent_example_observation, agent_example_state, agent_example_action = offline_replay_buffer.convert_to_critic_format(
    {
        "base_image": offline_replay_buffer.dataset_dict['base_image'][0][np.newaxis],
        "left_wrist_image": offline_replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
        "state": offline_replay_buffer.dataset_dict['state'][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict['actions'][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent = load_agent(
        seed=cfg.seed,
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
        resume=resuming,
        replan_steps=cfg.replan_steps,
        default_prompt=env.task_description,
        residual_action_xyzg=getattr(cfg, 'residual_action_xyzg', False),
    )
    
    start_step = 0
    if resuming:
        agent = restore_checkpoint(checkpoint_manager, agent)
        agent = agent.cache_infer_params()
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    # ── Critic pretraining (XQCfD-style critic/actor coherence warmup) ──────
    # Only on a fresh run — a resumed run's critic has already been through
    # this (or through real online training), so redoing it here would be
    # meaningless and could even undo real progress.
    #
    # Trains the critic ONLY (residual actor, base VLA, and temperature are
    # left untouched) on offline demo data, using the exact same
    # update_critic() logic as normal training — same argmax-over-candidates
    # target computation, same masking, nothing algorithmically different.
    # The point is purely to get the critic roughly "coherent" with the SFT
    # policy's own actions before the online loop starts pulling the residual
    # policy toward whatever a still-randomly-initialized critic prefers.
    critic_pretrain_steps = int(getattr(FLAGS.config, "critic_pretrain_steps", 0) or 0)
    if not resuming and model_cls in ("EXPOLearner", "EXPOLearnerOld") and critic_pretrain_steps > 0:
        # Demos live in offline_replay_buffer when offline_ratio > 0 (the
        # normal ExpoFT setup); BatchProcessor's constructor instead seeds
        # them into the online replay_buffer when offline_ratio == 0 — follow
        # whichever buffer actually received the dataset.
        pretrain_buffer = offline_replay_buffer if cfg.offline_ratio > 0 else replay_buffer
        logging.info(
            "Pretraining critic for %d steps on demo data (%s buffer)...",
            critic_pretrain_steps,
            "offline" if cfg.offline_ratio > 0 else "online (offline_ratio=0, demos live there instead)",
        )
        pretrain_iterator = pretrain_buffer.get_iterator(
            sample_args={"batch_size": cfg.batch_size},
            data_sharding=data_sharding,
        )
        # Give pretrain/* metrics their own independent step axis
        # ("pretrain_step"), decoupled from the default global step counter
        # that the main training loop below uses for training/* metrics (via
        # wandb.log(..., step=i)). Without this, wandb treats "step" as one
        # single global, monotonically-increasing counter shared by every
        # metric in the run regardless of name — logging pretraining at
        # steps 0..N-1 (or any other scheme) would collide with, and get
        # silently dropped against, whatever the main loop logs afterward
        # (this is what caused the "steps must be monotonically increasing"
        # warnings that silently dropped every single pretrain data point).
        # TensorBoard doesn't share this problem (each tag is an independent
        # scalar stream), so no equivalent change is needed for tb_writer.
        wandb.define_metric("pretrain_step")
        wandb.define_metric("pretrain/*", step_metric="pretrain_step")
        for pretrain_step in tqdm.tqdm(
            range(critic_pretrain_steps), desc="Critic pretraining", disable=not FLAGS.tqdm
        ):
            pretrain_batch = next(pretrain_iterator)
            pretrain_batch = pretrain_buffer.apply_data_sharding(pretrain_batch, data_sharding)
            # update_critic() expects a batch already run through the same two
            # steps _update_jit() normally applies before ever calling it —
            # augmentation, then prepare_critic_batch() (raw "image"/
            # "next_image" -> the structured "observations"/"next_observations"
            # format update_critic actually reads). Skipping these is what
            # crashed the first version of this loop with a bare
            # KeyError('next_observations').
            pretrain_batch = dict(pretrain_batch)
            rng, key1 = jax.random.split(agent.rng)
            rng, key2 = jax.random.split(rng)
            pretrain_batch["image"] = agent.data_augmentation_fn(key1, pretrain_batch["image"])
            pretrain_batch["next_image"] = agent.data_augmentation_fn(key2, pretrain_batch["next_image"])
            pretrain_batch = prepare_critic_batch(
                pretrain_batch,
                agent.actor.model_config.action_dim,
                agent.action_dim,
                agent.state_dim,
                agent.action_horizon,
                agent.replan_steps,
            )
            agent = agent.replace(rng=jax.device_put(rng, replicated_sharding))
            agent, pretrain_info = agent.update_critic(pretrain_batch)
            for k, v in pretrain_info.items():
                try:
                    tb_writer.add_scalar(f"pretrain/{k}", float(v), global_step=pretrain_step)
                except (TypeError, ValueError):
                    pass
            # No step= kwarg here on purpose — the custom step_metric wiring
            # above means wandb plots these against "pretrain_step" (logged
            # in the same call) instead of the shared global step counter.
            wandb.log({
                "pretrain_step": pretrain_step,
                **{f"pretrain/{k}": v for k, v in pretrain_info.items()},
            })
        logging.info("Critic pretraining complete.")

    # ── Actor behavior-cloning pretraining (PPO/GRPO only) ──────────────────
    # Same rationale as the critic pretraining above, but for a different
    # gap: PPOLearner/GRPOLearner's actor is a separate, randomly-initialized
    # TanhNormal network (+ its own batch_encoder) — the loaded VLA is used
    # only for input preprocessing / output denormalization, never to
    # initialize this network's own weights (see ppo.py/grpo.py's
    # load_agent() docstrings). Without this warm-start, on-policy PPO/GRPO
    # starts from a random policy and reproduces this project's own Phase 1
    # finding (on-policy from a random/pre-SFT start = 0% success, no
    # learning signal) despite intending to test on-policy FROM an SFT
    # checkpoint.
    #
    # Only on a fresh run, same as critic pretraining. Trains actor +
    # batch_encoder ONLY (never the value network — nothing SFT-relevant for
    # it to imitate) via maximum-likelihood behavior cloning on offline demo
    # data, using pretrain_actor_bc() — not part of the PPO/GRPO objective
    # itself, purely a warm-start executed before it.
    actor_pretrain_steps = int(getattr(FLAGS.config, "actor_pretrain_steps", 0) or 0)
    if not resuming and model_cls in ("PPOLearner", "GRPOLearner") and actor_pretrain_steps > 0:
        # For PPO/GRPO, offline_replay_buffer always holds the demo dataset
        # regardless of cfg.offline_ratio (which is forced to 0 for these
        # on-policy algos and doesn't reflect where the dataset was
        # inserted — see the is_on_policy_algo block above, which explicitly
        # seeds offline_replay_buffer for shape-inference purposes; that
        # same data serves this pretraining loop).
        actor_pretrain_buffer = offline_replay_buffer
        logging.info("Pretraining actor (behavior cloning) for %d steps on demo data...", actor_pretrain_steps)
        actor_pretrain_iterator = actor_pretrain_buffer.get_iterator(
            sample_args={"batch_size": cfg.batch_size},
            data_sharding=data_sharding,
        )
        wandb.define_metric("actor_pretrain_step")
        wandb.define_metric("actor_pretrain/*", step_metric="actor_pretrain_step")
        for actor_pretrain_step in tqdm.tqdm(
            range(actor_pretrain_steps), desc="Actor BC pretraining", disable=not FLAGS.tqdm
        ):
            actor_pretrain_batch = next(actor_pretrain_iterator)
            actor_pretrain_batch = actor_pretrain_buffer.apply_data_sharding(actor_pretrain_batch, data_sharding)
            actor_pretrain_batch = dict(actor_pretrain_batch)
            rng, key1 = jax.random.split(agent.rng)
            rng, key2 = jax.random.split(rng)
            actor_pretrain_batch["image"] = agent.data_augmentation_fn(key1, actor_pretrain_batch["image"])
            actor_pretrain_batch["next_image"] = agent.data_augmentation_fn(key2, actor_pretrain_batch["next_image"])
            actor_pretrain_batch = prepare_critic_batch(
                actor_pretrain_batch,
                agent.vla.model_config.action_dim,
                agent.action_dim,
                agent.state_dim,
                agent.action_horizon,
                agent.replan_steps,
            )
            agent = agent.replace(rng=jax.device_put(rng, replicated_sharding))
            agent, actor_pretrain_info = agent.pretrain_actor_bc(actor_pretrain_batch)
            for k, v in actor_pretrain_info.items():
                try:
                    tb_writer.add_scalar(f"actor_pretrain/{k}", float(v), global_step=actor_pretrain_step)
                except (TypeError, ValueError):
                    pass
            wandb.log({
                "actor_pretrain_step": actor_pretrain_step,
                **{f"actor_pretrain/{k}": v for k, v in actor_pretrain_info.items()},
            })
        logging.info("Actor BC pretraining complete.")

    episode_log = EpisodeState()
    training_log = TrainingStats(
        ep_count=replay_buffer.count_episodes_chronological() if resuming else 0,
    )
    logging.info("Resuming: ep_count set to %d (episodes in replay buffer).", training_log.ep_count)

    batch_processor.on_episode_start()

    dt = 1.0 / cfg.control_hz
    done = False
    env.reset()
    start_step_time = time.time()
    env.step(example_action.squeeze().tolist())
    action_plan = deque()
    action_type = "policy"
    episodes_since_update = 0
    combine_rng = jax.random.PRNGKey(cfg.seed + 100)

    def run_agent_updates(num_updates: int, metrics: dict):
        nonlocal agent, combine_rng
        for _ in range(num_updates):
            update_start = time.time()
            batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
            metrics["batch_info"] = get_batch_info(batch)
            agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
            agent, update_info = agent.update(agent, batch, cfg.utd_ratio, actor_batch)
            training_log.record_update_time(time.time() - update_start, metrics)
            for k, v in update_info.items():
                metrics[f"training/{k}"] = v

    for i in tqdm.tqdm(
        range(start_step, cfg.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        loop_start = time.time()
        step_metrics = {}

        observation = env.get_observation()
        # NOTE: done/success/reward/mask are deliberately NOT fetched here.
        # They must reflect the CONSEQUENCE of the action taken THIS
        # iteration (env.step() below), not the previous iteration's action
        # — fetching them here (before env.step()) was storing every
        # transition's reward/done/mask one step out of phase with its own
        # (observation, action) pair: transition i would get
        # (o_i, a_i, r_{i-1}, done_{i-1}) instead of (o_i, a_i, r_i, done_i).
        # This also delayed episode-boundary detection (the `if done:` reset
        # check below) by one step, letting one extra action execute past
        # the true terminal state before resetting.

        # Skip model inference while human is controlling.
        if not action_plan and action_type != "human":
            sample_start = time.time()
            action_chunk, agent, new_si = agent.sample_actions(observation)
            episode_log.sample_info_history.append(new_si)
            training_log.record_sample_time(time.time() - sample_start, step_metrics)
            action_plan.extend(action_chunk[:cfg.replan_steps])
        else:
            episode_log.sample_info_history.append(episode_log.sample_info_history[-1] if episode_log.sample_info_history else None)

        elapsed = time.time() - start_step_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

        has_action = bool(action_plan)
        action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
        real_action, action_type = env.step(action.tolist())
        start_step_time = time.time()
        # Fetch AFTER env.step(): now reflects the consequence of
        # `real_action` taken from `observation`, matching the (o_i, a_i,
        # r_i, done_i) convention the rest of the pipeline (Bellman backup,
        # GAE, ...) assumes.
        done, success, reward, mask = env.get_info_for_step()

        episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

        if action_type == "human":
            action_plan.clear()

        if has_action or action_type == "human":
            transition_dict = dict(
                observations=observation,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),
            )
            batch_processor.insert_transition(transition_dict)
        
        can_update = training_log.ep_count >= 10 and i >= cfg.batch_size
        if cfg.update_type == "step" and can_update:
            run_agent_updates(cfg.num_updates, step_metrics)

        if done:
            batch_processor.on_episode_done(success)
            env.reset()

            if cfg.update_type == "episode" and can_update:
                for _ in tqdm.tqdm(range(cfg.num_updates)):
                    run_agent_updates(1, step_metrics)
            elif cfg.update_type == "batch" and can_update:
                episodes_since_update += 1
                if episodes_since_update >= cfg.num_batch:
                    for _ in tqdm.tqdm(range(cfg.num_updates)):
                        run_agent_updates(1, step_metrics)
                    episodes_since_update = 0

            training_log.on_episode_done(episode_log, success, step_metrics)
            
            # Rolling success rate over last 20 episodes
            if not hasattr(training_log, '_success_window'):
                training_log._success_window = []
            training_log._success_window.append(float(success))
            if len(training_log._success_window) > 20:
                training_log._success_window.pop(0)
            step_metrics["eval/success_rate"] = np.mean(training_log._success_window)
            
            episode_log.reset()
            batch_processor.on_episode_start()

            observation = env.get_observation()
            done = False
            action_type = "policy"
            action_plan.clear()

        if cfg.checkpoint_model and cfg.checkpoint_interval > 0 and i > 0 and i % cfg.checkpoint_interval == 0:
            try:
                save_checkpoint(checkpoint_manager, agent, i)
                logging.info(f"Saved agent checkpoint at step {i} (interval={cfg.checkpoint_interval})")
            except Exception as e:
                logging.error(f"Could not save model checkpoint: {e}")

        if cfg.checkpoint_buffer and (has_action or action_type == "human"):
            try:
                save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=i)
            except Exception:
                logging.exception("Could not save agent buffer.")

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        
        # TensorBoard logging — convert to float() to handle JAX scalars
        # (jnp.float32 etc.) which don't pass isinstance(v, (int, float))
        for k, v in step_metrics.items():
            try:
                tb_writer.add_scalar(k, float(v), global_step=i)
            except (TypeError, ValueError):
                pass  # skip non-scalar values (e.g. batch_info dicts)
        wandb.log(step_metrics, step=i)
    
    if cfg.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, cfg.max_steps)
            logging.info(f"Saved final agent checkpoint at step {cfg.max_steps}")
        except Exception as e:
            logging.error(f"Could not save final checkpoint: {e}")
        logging.info("Waiting for checkpoint manager to finish")
        tb_writer.close()
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
