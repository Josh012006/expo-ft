#!/usr/bin/env python

from __future__ import annotations

import logging
import os
import time
from collections import deque

import etils.epath as epath
import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from expo_ft.agents import initialize_checkpoint_dir
from expo_ft.data.replay_buffer import create_replay_buffer
from expo_ft.env.env_client import EnvClientWrapper
from expo_ft.env.droid_utils import process_droid_dataset

import openpi.training.sharding as openpi_sharding

config_flags.DEFINE_config_file(
    "config",
    "configs/model/expo_ft_pi_config.py",
    "Training config (must match the checkpoint).",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_task",
    "configs/task/pick.py",
    "Task config (must match training).",
    lock_config=False,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "", "Path to DROID dataset (for example_action).")
flags.DEFINE_integer("num_data", 1, "Number of episodes to load from dataset (only need 1 for example_action).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory (e.g. .../checkpoints/<run_name>/checkpoints).")
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step to load; default is latest.")
flags.DEFINE_string("client_host", "localhost", "Rollout server host.")
flags.DEFINE_integer("client_port", 8102, "Rollout server port.")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("replan_steps", 8, "Replan every N steps (match training).")
flags.DEFINE_boolean("only_base_actions", False, "Use only base (OpenPI) actions, no residual, sample 1.")
flags.DEFINE_boolean("save_video", True, "Save evaluation videos.")
flags.DEFINE_integer("fsdp_devices", 1, "Number of FSDP devices (match training).")


def main(_):
    config = FLAGS.config
    config_task = FLAGS.config_task

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    if config_task.env_type != "droid":
        raise ValueError("This script is for DROID evaluation only; config_task.env_type must be 'droid'.")

    if not FLAGS.dataset_path or not FLAGS.checkpoint_dir:
        raise ValueError("--dataset_path and --checkpoint_dir are required.")

    checkpoint_dir_path = epath.Path(FLAGS.checkpoint_dir)
    if not checkpoint_dir_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir_path}")
    checkpoint_manager, _ = initialize_checkpoint_dir(
        checkpoint_dir_path,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    checkpoint_steps = tuple(checkpoint_manager.all_steps())
    step = FLAGS.checkpoint_step
    if step is None:
        step = max(checkpoint_steps) if checkpoint_steps else 0
        logger.info("Using latest checkpoint step %s", step)
    if step != 0:
        if step not in checkpoint_steps:
            raise ValueError(f"Step {step} not in checkpoint steps {checkpoint_steps}")
        logger.info("Will load checkpoint at step %s", step)

    example_action = config_task.example_action
    # Dataset only for agent observation/action shapes (run one sample through transform)
    dataset = process_droid_dataset(
        FLAGS.dataset_path,
        config_task,
        num_data=FLAGS.num_data,
    )

    task_description = config_task.language_instruction
    max_traj_len = config_task.auto_reset_steps
    dt = 1.0 / config_task.control_hz

    # Agent config (match train_pi_robo)
    mesh = openpi_sharding.make_mesh(FLAGS.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    model_cls = config.model_cls
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint
    else:
        raise ValueError(f"Unsupported model class: {model_cls}")

    from expo_ft.agents.vla.pi05 import build_pi05
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        config, FLAGS.seed, mesh, data_sharding, replicated_sharding,
        resume=(step != 0), default_prompt=task_description,
    )

    replay_buffer = create_replay_buffer(
        config=config,
        example_action=example_action,
        capacity=max_traj_len * 2,
        task_description=task_description,
        replan_steps=FLAGS.replan_steps,
        seed=FLAGS.seed,
    )
    replay_buffer.insert_dataset(dataset[:1])

    agent_example_observation, agent_example_state, agent_example_action = replay_buffer.convert_to_critic_format({
        "base_image": replay_buffer.dataset_dict["base_image"][0][np.newaxis],
        "left_wrist_image": replay_buffer.dataset_dict["left_wrist_image"][0][np.newaxis],
        "state": replay_buffer.dataset_dict["state"][0][np.newaxis],
        "actions": replay_buffer.dataset_dict["actions"][0][np.newaxis],
    })
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]
    agent = load_agent(
        seed=FLAGS.seed,
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
        resume=(step != 0),
        replan_steps=FLAGS.replan_steps,
        default_prompt=task_description,
        residual_action_xyzg=config_task.residual_action_xyzg,
    )

    if step != 0:
        agent = restore_checkpoint(checkpoint_manager, agent, step=step)
        logger.info("Loaded checkpoint at step %s", step)

    if hasattr(agent, 'cache_infer_params'):
        agent = agent.cache_infer_params()

    video_dir = None
    if FLAGS.save_video:
        save_video_dir = os.path.join(
            os.path.dirname(FLAGS.checkpoint_dir), "eval", f"step_{step}"
        )
        parts = []
        if FLAGS.only_base_actions:
            parts.append("only_base")
        subdir = "_".join(parts) if parts else "full"
        video_dir = os.path.join(save_video_dir, subdir)
        os.makedirs(video_dir, exist_ok=True)
        logger.info("Saving evaluation videos to %s", video_dir)

    eval_env_creation_request = {
        "example_action": example_action,
        "env_usage": "eval",
        "video_dir": video_dir or "",
    }
    logger.info("Connecting to rollout server at %s:%s ...", FLAGS.client_host, FLAGS.client_port)
    env = EnvClientWrapper(
        env_creation_request=eval_env_creation_request,
        host=FLAGS.client_host,
        port=FLAGS.client_port,
    )
    print("resetting environment...")
    env.reset()
    print("environment reset")

    time.sleep(10)

    successes = []
    episode_returns = []
    episode_lengths = []

    for ep in range(FLAGS.num_episodes):
        logger.info("Episode %d / %d", ep + 1, FLAGS.num_episodes)
        observation = env.reset()
        start_time = time.time()
        action_plan = deque()
        sample_info_history = []
        ep_return = 0.0
        ep_len = 0

        for step in range(max_traj_len):
            step_t0 = time.time()
            timing = {
                "wait_ms": 0.0,
                "obs_ms": 0.0,
                "info_ms": 0.0,
                "plan_ms": 0.0,
                "act_ms": 0.0,
            }

            t_obs0 = time.time()
            observation = env.get_observation()
            timing["obs_ms"] = (time.time() - t_obs0) * 1000.0
            t_info0 = time.time()
            done, success, reward, _ = env.get_info_for_step()
            timing["info_ms"] = (time.time() - t_info0) * 1000.0

            t_plan0 = time.time()
            if not action_plan:
                action_chunk, agent, new_si = agent.sample_actions(
                    observation,
                    only_base_actions=FLAGS.only_base_actions,
                )
                action_chunk = np.asarray(jax.device_get(action_chunk))
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[None, :]
                action_plan.extend(list(action_chunk[: FLAGS.replan_steps]))
                sample_info_history.append(new_si)
            else:
                sample_info_history.append(sample_info_history[-1] if sample_info_history else None)
            timing["plan_ms"] = (time.time() - t_plan0) * 1000.0
            action = action_plan.popleft()

            ep_return += reward
            ep_len += 1

            if done:
                timing_total_ms = (time.time() - step_t0) * 1000.0
                logger.info(
                    "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
                    ep + 1,
                    step,
                    timing_total_ms,
                    timing["wait_ms"],
                    timing["obs_ms"],
                    timing["info_ms"],
                    timing["plan_ms"],
                    timing["act_ms"],
                    done,
                )
                break

            # Match DROID collect loop timing: wait from previous step end.
            elapsed = time.time() - start_time
            sleep_left = dt - elapsed
            if sleep_left > 0:
                t_wait0 = time.time()
                time.sleep(sleep_left)
                timing["wait_ms"] = (time.time() - t_wait0) * 1000.0

            t_act0 = time.time()
            env.step(np.asarray(action).tolist())
            timing["act_ms"] = (time.time() - t_act0) * 1000.0
            start_time = time.time()

            timing_total_ms = (time.time() - step_t0) * 1000.0
            logger.info(
                "[timing][ep %d step %d] total=%.1fms wait=%.1f obs=%.1f info=%.1f plan=%.1f act=%.1f done=%s",
                ep + 1,
                step,
                timing_total_ms,
                timing["wait_ms"],
                timing["obs_ms"],
                timing["info_ms"],
                timing["plan_ms"],
                timing["act_ms"],
                done,
            )

        successes.append(success)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        logger.info("  success=%s return=%.1f len=%d", success, ep_return, ep_len)

    n = len(successes)
    success_rate = float(np.mean(successes))
    mean_return = float(np.mean(episode_returns))
    mean_len = float(np.mean(episode_lengths))
    logger.info("Evaluation complete: success_rate=%.2f (%d/%d) mean_return=%.2f mean_len=%.1f",
                success_rate, int(np.sum(successes)), n, mean_return, mean_len)
    print(f"success_rate={success_rate:.2f} mean_return={mean_return:.2f} mean_len={mean_len:.1f}")


if __name__ == "__main__":
    app.run(main)
