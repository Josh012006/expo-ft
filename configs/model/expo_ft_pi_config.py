from ml_collections.config_dict import config_dict
from configs.model import sac_config


def get_config():
    config = sac_config.get_config()

    config.model_cls = "EXPOLearner"

    # NOTE: num_qs/num_min_qs/critic_layer_norm below are dead for EXPOLearner
    # as of the categorical-critic rewrite (expo_ft.py) — kept here only
    # because SACLearner (sac.py, which also extends this same base config)
    # still uses the old ensemble-of-scalars critic architecture and reads
    # these fields directly. Do not remove.
    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    # Categorical (C51-style, bounded support) critic — EXPOLearner only, per
    # XQC (arXiv 2509.25174) / XQCfD (arXiv 2605.10734). v_min/v_max apply to
    # NORMALIZED reward units (rewards are divided by a running RMS estimate
    # before the Bellman projection — see reward_scale_decay and
    # expo_ft.py's update_critic()), so they should be domain-agnostic and
    # NOT need per-task hand-tuning. Still watch target_q_max/min (the
    # normalized ones, not the _denorm logging variants): if Q is pinned at
    # exactly v_min or v_max for a meaningful fraction of training even in
    # normalized units, the support itself is too narrow and needs widening.
    config.num_atoms = 101
    config.v_min = -10.0
    config.v_max = 20.0
    config.reward_scale_decay = 0.99  # EMA decay for the running reward-RMS estimate; higher = slower-adapting, more stable

    # XQCfD-style KL regularization for the edit/residual policy (replaces
    # the generic entropy bonus with a penalty for deviating from a fixed
    # N(0, kl_ref_std) reference in pre-tanh space, when enabled).
    # 0.0 = disabled (default, matches pre-existing entropy-only behavior).
    config.kl_coef = 0.0
    config.kl_ref_std = 1.0
    config.critic_hidden_dims = (512, 512, 512, 512)

    config.N = 8
    config.n_edit_samples = 8

    config.adjust_target_entropy = False
    config.entropy_scale = 1.0
    config.edit_scale = 0.2
    config.fixed_temperature = config_dict.placeholder(float)  # if set, bypasses the learned SAC temperature entirely
    config.critic_grad_clip_norm = config_dict.placeholder(float)  # if set, clips critic + encoder grads to this global norm before adam/adamw
    config.critic_pretrain_steps = 0  # if >0, run this many critic-only update_critic() steps on offline demo data before RL starts (XQCfD-style critic/actor coherence warmup). 0 = disabled (default, matches pre-existing behavior).
    config.actor_drop = 0.0
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.latent_dim_image = 512
    config.latent_dim_state = 64
    config.include_state = True
    config.encoder_stage_sizes = (3, 4, 6, 3)
    config.encoder_num_filters = 64
    config.hidden_dims = (256, 256, 256)

    config.encode_batch_split = 1
    config.batch_split = 1

    config.use_pi05 = True
    config.pi05_config_name = "expo_pi05_droid_lora_finetune_sft_cartesian_state"
    config.pi05_resize_size = 224
    config.freeze_pi05_encoder = True
    config.freeze_critic_encoder = False  # if True, encoder is frozen for Q (only extract embeddings)
    
    config.pi05_weight_loader_path = "" # pi05 sft checkpoint path
    # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id.
    config.pi05_assets_dir = ""
    config.pi05_asset_id = ""
    config.actor_success_only = True
    config.use_full_augmentation = True  # False = only crop (no rotate/color jitter)

    return config
