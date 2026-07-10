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

    # Override FLAGS.config RL hyperparameters from the task YAML so everything
    # is configured in one place (the YAML) rather than split between YAML and
    # configs/model/expo_ft_pi_config.py.
    # NOTE: float() wrapping below is a deliberate defense against a PyYAML quirk —
    # bare scientific notation without a decimal point (e.g. "3e-4") is parsed as
    # a STRING, not a float (needs "3.0e-4" to parse correctly). ml_collections
    # then raises a TypeError trying to assign a str into a float-typed field.
    # float(x) is a no-op if x is already a float, and fixes it if x is a
    # not-quite-valid-YAML-float string — belt and suspenders alongside fixing
    # the YAML values themselves.
    FLAGS.config.actor_lr         = float(getattr(cfg, "rl_lr", FLAGS.config.actor_lr))
    FLAGS.config.critic_lr        = float(getattr(cfg, "rl_lr", FLAGS.config.critic_lr))
    FLAGS.config.discount         = float(getattr(cfg, "rl_discount", FLAGS.config.discount))
    FLAGS.config.tau              = float(getattr(cfg, "rl_tau", FLAGS.config.tau))
    FLAGS.config.init_temperature = float(getattr(cfg, "rl_init_temperature", FLAGS.config.init_temperature))
    FLAGS.config.adjust_target_entropy = getattr(cfg, "rl_adjust_target_entropy", FLAGS.config.adjust_target_entropy)
    _rl_fixed_temperature = getattr(cfg, "rl_fixed_temperature", FLAGS.config.fixed_temperature)
    FLAGS.config.fixed_temperature = float(_rl_fixed_temperature) if _rl_fixed_temperature is not None else None
    if hasattr(cfg, "rl_hidden_dims"):
        FLAGS.config.hidden_dims  = tuple(cfg.rl_hidden_dims)
    FLAGS.config.edit_scale       = float(getattr(cfg, "rl_edit_scale", FLAGS.config.edit_scale))
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

    assert 0.0 <= cfg.offline_ratio <= 1.0

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

    model_cls = FLAGS.config.model_cls
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

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
        offline_ratio=cfg.offline_ratio,
        actor_success_only=actor_success_only,
        use_dagger_hil_sampling=use_dagger_hil_sampling,
        dataset=dataset,
    )

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
        done, success, reward, mask = env.get_info_for_step()

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
