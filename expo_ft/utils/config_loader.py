"""
Load task config from a YAML file and expose it as a simple namespace object.
Used as a drop-in replacement for ml_collections config_flags in train_pi_robo.py.
"""

import os
import yaml
from datetime import datetime
from types import SimpleNamespace


def load_task_config(yaml_path: str) -> SimpleNamespace:
    """
    Load task config from YAML and return a SimpleNamespace object.
    Automatically resolves paths relative to the repo root.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = SimpleNamespace(**raw)

    # Compute max_steps from episodes config
    if hasattr(cfg, 'max_episodes') and hasattr(cfg, 'max_steps_per_episode'):
        cfg.max_steps = cfg.max_episodes * cfg.max_steps_per_episode

    # Convert nested lists to tuples where needed (e.g. rl_hidden_dims)
    if hasattr(cfg, "rl_hidden_dims"):
        cfg.rl_hidden_dims = tuple(cfg.rl_hidden_dims)

    return cfg


def apply_task_overrides(config, cfg: SimpleNamespace):
    """Write the task YAML `cfg` into the model config `config` (ml_collections ConfigDict of configs/model/*.py).

    THE single place where the YAML reaches the model config. Called by every entry point that builds an agent
    (train_pi_robo.py, scripts/eval_policy.py, scripts/compare_argmax_vs_reference_q.py,
    scripts/validate_rl_action_chunks.py): a script that skipped it used the code defaults instead of the values the
    run was trained with (architecture fields such as rl_hidden_dims / rl_edit_scale / rl_N decide whether an RL
    checkpoint even restores). It was the body of train_pi_robo.py's main(), moved here unchanged except for the last
    two steps (actor_success_only, tactile / force).

    Order: openpi config name -> hyperparameters of the learner `config.model_cls` -> actor_success_only -> tactile / force.
    Returns the AuxSpec in force (see expo_ft/utils/aux_features.py).
    """
    config.pi05_config_name = get_sft_config_name(cfg)
    config.skip_repack_transforms = cfg.skip_repack_transforms
    model_cls = config.model_cls

    # Override config RL hyperparameters from the task YAML so everything
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
    if model_cls in ("EXPOLearner", "EXPOLearnerCategorical"):
        # --- unchanged from before this refactor: byte-for-byte identical ---
        config.actor_lr         = float(getattr(cfg, "rl_lr", config.actor_lr))
        config.critic_lr        = float(getattr(cfg, "rl_lr", config.critic_lr))
        # NOTE: previously orphaned -- temp_lr silently stayed at whatever
        # config.temp_lr (3e-4, from sac_config.py) already was,
        # decoupled from rl_lr above and with no YAML field to override it
        # independently. Wired here the same way as the other rl_* fields;
        # default is unchanged (falls back to config.temp_lr) for any
        # YAML that doesn't set rl_temp_lr yet.
        config.temp_lr          = float(getattr(cfg, "rl_temp_lr", config.temp_lr))
        config.discount         = float(getattr(cfg, "rl_discount", config.discount))
        config.tau              = float(getattr(cfg, "rl_tau", config.tau))
        config.init_temperature = float(getattr(cfg, "rl_init_temperature", config.init_temperature))
        config.adjust_target_entropy = getattr(cfg, "rl_adjust_target_entropy", config.adjust_target_entropy)
        _rl_fixed_temperature = getattr(cfg, "rl_fixed_temperature", config.fixed_temperature)
        config.fixed_temperature = float(_rl_fixed_temperature) if _rl_fixed_temperature is not None else None
        _rl_critic_weight_decay = getattr(cfg, "rl_critic_weight_decay", config.critic_weight_decay)
        config.critic_weight_decay = float(_rl_critic_weight_decay) if _rl_critic_weight_decay is not None else None
        _rl_critic_grad_clip_norm = getattr(cfg, "rl_critic_grad_clip_norm", config.critic_grad_clip_norm)
        config.critic_grad_clip_norm = float(_rl_critic_grad_clip_norm) if _rl_critic_grad_clip_norm is not None else None
        config.freeze_critic_encoder = getattr(cfg, "rl_freeze_critic_encoder", config.freeze_critic_encoder)
        if hasattr(cfg, "rl_hidden_dims"):
            config.hidden_dims  = tuple(cfg.rl_hidden_dims)
        config.edit_scale       = float(getattr(cfg, "rl_edit_scale", config.edit_scale))
        config.N = int(getattr(cfg, "rl_N", config.N))
        config.n_edit_samples = int(getattr(cfg, "rl_n_edit_samples", config.n_edit_samples))
        # --- end of original ExpoFT block; critic_pretrain_steps added below is
        # a new, default-off (0) field — behavior for existing configs that
        # don't set rl_critic_pretrain_steps is unchanged ---
        config.critic_pretrain_steps = int(getattr(cfg, "rl_critic_pretrain_steps", config.critic_pretrain_steps))
        config.actor_bc_pretrain_steps = int(getattr(cfg, "rl_actor_bc_pretrain_steps", config.actor_bc_pretrain_steps))
        config.num_atoms = int(getattr(cfg, "rl_num_atoms", config.num_atoms))
        config.v_min = float(getattr(cfg, "rl_v_min", config.v_min))
        config.v_max = float(getattr(cfg, "rl_v_max", config.v_max))
        config.reward_scale_decay = float(getattr(cfg, "rl_reward_scale_decay", config.reward_scale_decay))
        config.use_reward_normalization = bool(getattr(cfg, "rl_use_reward_normalization", config.use_reward_normalization))
        config.kl_coef = float(getattr(cfg, "rl_kl_coef", config.kl_coef))
        config.entropy_scale = float(getattr(cfg, "rl_entropy_scale", config.entropy_scale))
        config.kl_ref_std = float(getattr(cfg, "rl_kl_ref_std", config.kl_ref_std))
        config.use_hetstat_policy = bool(getattr(cfg, "rl_use_hetstat_policy", config.use_hetstat_policy))
        config.hetstat_num_rff_features = int(getattr(cfg, "rl_hetstat_num_rff_features", config.hetstat_num_rff_features))
        config.hetstat_var_lr_multiplier = float(getattr(cfg, "rl_hetstat_var_lr_multiplier", config.hetstat_var_lr_multiplier))
        config.use_double_q_selection = bool(getattr(cfg, "rl_use_double_q_selection", config.use_double_q_selection))
        config.use_clipped_double_q = bool(getattr(cfg, "rl_use_clipped_double_q", config.use_clipped_double_q))
    elif model_cls == "PPOLearner":
        config.actor_lr  = float(getattr(cfg, "ppo_lr", config.actor_lr))
        config.critic_lr = float(getattr(cfg, "ppo_lr", config.critic_lr))
        config.discount  = float(getattr(cfg, "ppo_discount", config.discount))
        config.gae_lambda        = float(getattr(cfg, "ppo_gae_lambda", config.gae_lambda))
        config.clip_eps          = float(getattr(cfg, "ppo_clip_eps", config.clip_eps))
        config.value_loss_coef   = float(getattr(cfg, "ppo_value_loss_coef", config.value_loss_coef))
        config.entropy_coef      = float(getattr(cfg, "ppo_entropy_coef", config.entropy_coef))
        _ppo_value_clip_eps = getattr(cfg, "ppo_value_clip_eps", config.value_clip_eps)
        config.value_clip_eps = float(_ppo_value_clip_eps) if _ppo_value_clip_eps is not None else None
        _ppo_max_grad_norm = getattr(cfg, "ppo_max_grad_norm", config.max_grad_norm)
        config.max_grad_norm = float(_ppo_max_grad_norm) if _ppo_max_grad_norm is not None else None
        config.num_minibatches = int(getattr(cfg, "ppo_num_minibatches", config.num_minibatches))
        if hasattr(cfg, "ppo_hidden_dims"):
            config.hidden_dims = tuple(cfg.ppo_hidden_dims)
        config.actor_pretrain_steps = int(getattr(cfg, "ppo_actor_pretrain_steps", config.actor_pretrain_steps))
        config.rollout_length = int(getattr(cfg, "ppo_rollout_length", config.rollout_length))
        config.actor_log_std_min = float(getattr(cfg, "ppo_actor_log_std_min", config.actor_log_std_min))
    elif model_cls == "GRPOLearner":
        config.actor_lr     = float(getattr(cfg, "grpo_lr", config.actor_lr))
        config.group_size   = int(getattr(cfg, "grpo_group_size", config.group_size))
        config.clip_eps     = float(getattr(cfg, "grpo_clip_eps", config.clip_eps))
        config.kl_coef      = float(getattr(cfg, "grpo_kl_coef", config.kl_coef))
        config.entropy_coef = float(getattr(cfg, "grpo_entropy_coef", config.entropy_coef))
        _grpo_max_grad_norm = getattr(cfg, "grpo_max_grad_norm", config.max_grad_norm)
        config.max_grad_norm = float(_grpo_max_grad_norm) if _grpo_max_grad_norm is not None else None
        config.num_minibatches = int(getattr(cfg, "grpo_num_minibatches", config.num_minibatches))
        if hasattr(cfg, "grpo_hidden_dims"):
            config.hidden_dims = tuple(cfg.grpo_hidden_dims)
        config.actor_pretrain_steps = int(getattr(cfg, "grpo_actor_pretrain_steps", config.actor_pretrain_steps))
        config.rollout_length = int(getattr(cfg, "grpo_rollout_length", config.rollout_length))
        config.actor_log_std_min = float(getattr(cfg, "grpo_actor_log_std_min", config.actor_log_std_min))
    elif model_cls == "PPOResidualLearner":
        # Same fields as PPOLearner's own override block above, EXCEPT
        # actor_pretrain_steps -- there is no pretrain_actor_bc on this
        # class (no random-init-actor problem to warm-start away, see
        # ppo_residual_pi_config.py's docstring), so that field does not
        # exist on this config at all; setting it would error. edit_scale
        # is new here (unused by plain PPOLearner).
        config.actor_lr  = float(getattr(cfg, "ppo_lr", config.actor_lr))
        config.critic_lr = float(getattr(cfg, "ppo_lr", config.critic_lr))
        config.discount  = float(getattr(cfg, "ppo_discount", config.discount))
        config.gae_lambda        = float(getattr(cfg, "ppo_gae_lambda", config.gae_lambda))
        config.clip_eps          = float(getattr(cfg, "ppo_clip_eps", config.clip_eps))
        config.value_loss_coef   = float(getattr(cfg, "ppo_value_loss_coef", config.value_loss_coef))
        config.entropy_coef      = float(getattr(cfg, "ppo_entropy_coef", config.entropy_coef))
        _ppo_value_clip_eps = getattr(cfg, "ppo_value_clip_eps", config.value_clip_eps)
        config.value_clip_eps = float(_ppo_value_clip_eps) if _ppo_value_clip_eps is not None else None
        _ppo_max_grad_norm = getattr(cfg, "ppo_max_grad_norm", config.max_grad_norm)
        config.max_grad_norm = float(_ppo_max_grad_norm) if _ppo_max_grad_norm is not None else None
        config.num_minibatches = int(getattr(cfg, "ppo_num_minibatches", config.num_minibatches))
        if hasattr(cfg, "ppo_hidden_dims"):
            config.hidden_dims = tuple(cfg.ppo_hidden_dims)
        config.rollout_length = int(getattr(cfg, "ppo_rollout_length", config.rollout_length))
        config.actor_log_std_min = float(getattr(cfg, "ppo_actor_log_std_min", config.actor_log_std_min))
        config.edit_scale = float(getattr(cfg, "ppo_edit_scale", config.edit_scale))
    elif model_cls == "GRPOResidualLearner":
        # Same fields as GRPOLearner's own override block above, EXCEPT
        # actor_pretrain_steps (same rationale as PPOResidualLearner's
        # branch just above), plus edit_scale.
        config.actor_lr     = float(getattr(cfg, "grpo_lr", config.actor_lr))
        config.group_size   = int(getattr(cfg, "grpo_group_size", config.group_size))
        config.clip_eps     = float(getattr(cfg, "grpo_clip_eps", config.clip_eps))
        config.kl_coef      = float(getattr(cfg, "grpo_kl_coef", config.kl_coef))
        config.entropy_coef = float(getattr(cfg, "grpo_entropy_coef", config.entropy_coef))
        _grpo_max_grad_norm = getattr(cfg, "grpo_max_grad_norm", config.max_grad_norm)
        config.max_grad_norm = float(_grpo_max_grad_norm) if _grpo_max_grad_norm is not None else None
        config.num_minibatches = int(getattr(cfg, "grpo_num_minibatches", config.num_minibatches))
        if hasattr(cfg, "grpo_hidden_dims"):
            config.hidden_dims = tuple(cfg.grpo_hidden_dims)
        config.rollout_length = int(getattr(cfg, "grpo_rollout_length", config.rollout_length))
        config.actor_log_std_min = float(getattr(cfg, "grpo_actor_log_std_min", config.actor_log_std_min))
        config.edit_scale = float(getattr(cfg, "grpo_edit_scale", config.edit_scale))
    elif model_cls == "SACLearner":
        config.actor_lr  = float(getattr(cfg, "sac_lr", config.actor_lr))
        config.critic_lr = float(getattr(cfg, "sac_lr", config.critic_lr))
        config.discount  = float(getattr(cfg, "sac_discount", config.discount))
        config.tau       = float(getattr(cfg, "sac_tau", config.tau))
        config.init_temperature = float(getattr(cfg, "sac_init_temperature", config.init_temperature))
        _sac_target_entropy = getattr(cfg, "sac_target_entropy", config.target_entropy)
        config.target_entropy = float(_sac_target_entropy) if _sac_target_entropy is not None else None
        _sac_critic_weight_decay = getattr(cfg, "sac_critic_weight_decay", config.critic_weight_decay)
        config.critic_weight_decay = float(_sac_critic_weight_decay) if _sac_critic_weight_decay is not None else None
        config.num_qs = int(getattr(cfg, "sac_num_qs", config.num_qs))
        if hasattr(cfg, "sac_hidden_dims"):
            config.hidden_dims = tuple(cfg.sac_hidden_dims)
    # BCLearner: no RL hyperparameter overrides needed — imitation-only,
    # no critic/GAE/advantage machinery to tune here.

    # Sync actor_success_only from the task YAML into the model config too —
    # BatchProcessor already reads it from cfg (via train_pi_robo's own actor_success_only variable), but the
    # EXPOLearner agent itself reads its own copy from the model config, which defaults to True in
    # expo_ft_pi_config.py. Without this sync, BatchProcessor (correctly seeing the YAML's actor_success_only=False)
    # never builds an actor_batch, while the agent (still seeing the config's default True) expects one — crash.
    config.actor_success_only = getattr(cfg, "actor_success_only", False)

    return apply_aux_overrides(config, cfg)


def apply_aux_overrides(config, cfg: SimpleNamespace):
    """Tactile / force input of the residual actor and the critic: YAML fields use_tactile, use_force, force_mode ->
    the same three fields of the model config, where the replay buffer (create_replay_buffer) and the agent
    (EXPOLearner.create, through agent_kwargs) read them. Returns the AuxSpec.

    Refuses instead of ignoring: the signals are implemented for model_cls EXPOLearner only. Another learner would
    train without them while the YAML says otherwise.
    """
    from expo_ft.utils.aux_features import AuxSpec
    spec = AuxSpec.from_config(cfg)          # validates the three fields (types, force_mode, proxy not implemented)
    model_cls = config.model_cls
    if spec.enabled and model_cls != "EXPOLearner":
        raise ValueError(
            f"use_tactile / use_force ({spec.describe()}) are only implemented for model_cls EXPOLearner, "
            f"got {model_cls}. Set both to false, or train with EXPOLearner."
        )
    if model_cls == "EXPOLearner":
        for name, value in spec.to_config_kwargs().items():
            setattr(config, name, value)
    return spec


def check_rl_action_space_supported(cfg: SimpleNamespace) -> None:
    """Refuse a task whose model output is not directly the action sent to the environment.

    `chunk_action_reference: chunk_start_state` (FORGE, pi05_droid_jointpos): the model outputs a chunk of OFFSETS from
    the joint state at plan time and the client adds that state back (scripts/eval_policy.py). The RL loop
    (train_pi_robo.py) has no such step, and its replay buffer stores the executed actions in the frame the model does not
    use: a run would send offsets to the robot as absolute joint targets and train the residual actor and the critic on
    actions of two different frames. Nothing in this repository converts them yet, so stop here, before any GPU time is
    spent, rather than train on garbage. Evaluation (scripts/eval_policy.py) handles the offsets and is not affected.
    """
    reference = getattr(cfg, "chunk_action_reference", None)
    if reference:
        raise NotImplementedError(
            f"chunk_action_reference: {reference!r} is set in the task YAML, and RL training does not handle it yet "
            f"(the model's chunk is made of offsets from the plan-time joint state, the environment expects absolute "
            f"joint targets, and the replay buffer would mix both frames). Remove it only once train_pi_robo.py and "
            f"PiReplayBuffer convert between the two."
        )


def resolve_run_dir(cfg: SimpleNamespace, resume_dir: str | None = None, suffix: str | None = None) -> tuple[str, bool]:
    """
    Resolve the run directory for this experiment.

    - If `resume_dir` is set: resume from that exact directory.
    - Otherwise: create a new timestamped subdirectory under cfg.output_dir,
      optionally with `suffix` appended after the timestamp (e.g. "rl", so RL
      runs are visually distinguishable from SFT runs at a glance).

    `resume_dir` is passed in explicitly (rather than read from a single
    shared cfg.resume_dir) because SFT and RL each resume independently —
    callers should pass cfg.sft_resume_dir or cfg.rl_resume_dir as appropriate.

    Returns:
        (run_dir, resuming) where resuming is True if we are resuming.
    """
    if resume_dir is not None:
        run_dir = resume_dir
        resuming = True
    else:
        # Second-level timestamp alone isn't unique enough: two jobs launched
        # in the same wall-clock second (e.g. submitting an old-critic run
        # and a new-critic run back to back for comparison) get an IDENTICAL
        # run_dir, and os.makedirs(..., exist_ok=True) silently lets both
        # write into the same checkpoint folder -- causing concurrent-write
        # races in orbax's OCDBT backend (manifest.ocdbt rename/ENOENT
        # errors). Disambiguate with the SLURM job ID when present (no two
        # jobs ever share one, and it's also the most useful thing to see in
        # a directory listing); fall back to microsecond precision locally.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        job_id = os.environ.get("SLURM_JOB_ID")
        disambiguator = job_id if job_id else datetime.now().strftime("%f")
        run_name = f"{cfg.run_name}_{timestamp}-{disambiguator}"
        if suffix:
            run_name = f"{run_name}_{suffix}"
        run_dir = os.path.join(cfg.output_dir, run_name)
        resuming = False

    os.makedirs(run_dir, exist_ok=True)
    return run_dir, resuming


def get_sft_config_name(cfg: SimpleNamespace) -> str:
    """
    Return the openpi TrainConfig name to use for SFT warmup,
    based on task config flags.
    """
    use_cartesian = getattr(cfg, 'use_cartesian_state', True)
    if cfg.sft_use_lora:
        if use_cartesian:
            return "expo_pi05_droid_lora_finetune_sft_cartesian_state"
        else:
            return "expo_pi05_droid_lora_finetune_sft_joint_state"
    else:
        if use_cartesian:
            return "expo_pi05_droid_full_finetune_sft_cartesian_state"
        else:
            return "expo_pi05_droid_full_finetune_sft_joint_state"


def get_sft_train_config_name(cfg: SimpleNamespace) -> str:
    """
    openpi TrainConfig used by the SFT *training* stage only.

    A task YAML can force it with `sft_train_config_name` (FORGE: the "..._joint_state_delta" config, which
    turns the demos' absolute joint targets into chunk offsets at training time). Without that key this is
    get_sft_config_name(cfg), i.e. unchanged for the ManiSkill tasks.

    Everything else (eval, norm_stats, RL) keeps calling get_sft_config_name(), so a checkpoint trained with the
    delta config is still loaded with the plain "..._joint_state" config: same model, the offsets are
    handled by the client (chunk_action_reference: chunk_start_state).
    """
    return getattr(cfg, "sft_train_config_name", None) or get_sft_config_name(cfg)
