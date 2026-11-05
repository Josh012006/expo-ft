#! /usr/bin/env python
import os
import gc
import json
import logging
import time
from collections import deque
from pathlib import Path

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
from expo_ft.env.droid_utils import load_demo_dataset
from expo_ft.utils.log_utils import EpisodeState, TrainingStats
from expo_ft.utils.train_utils import get_batch_info, init_logging, init_wandb
from expo_ft.utils.action_space import chunk_to_env_actions
from expo_ft.utils.config_loader import apply_task_overrides, load_task_config, resolve_run_dir


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


def get_or_create_episode_seeds(output_dir, n_episodes, master_seed):
    """Mirrors scripts/eval_curve.py's own get_or_create_episode_seeds()
    EXACTLY (same generation logic: np.random.default_rng(master_seed),
    same cache file name/location) so a periodic in-training eval here uses
    the IDENTICAL fixed seed list eval_curve.py would generate for the SAME
    (output_dir, n_episodes, master_seed) -- true apples-to-apples
    comparability with the SFT baseline's own evaluation, without needing to
    locate or copy any specific file: same inputs, same deterministic PRNG,
    same seeds, wherever this is called from. Kept as a separate copy here
    (not imported from scripts/eval_curve.py) to avoid a fragile
    cross-directory import; if eval_curve.py's own version ever changes,
    this one must be updated to match, or the two would silently diverge."""
    output_dir = Path(output_dir)
    seeds_path = output_dir / "episode_seeds.json"
    if seeds_path.exists():
        with open(seeds_path) as f:
            seeds = json.load(f)
        if len(seeds) != n_episodes:
            raise ValueError(
                f"Existing {seeds_path} has {len(seeds)} seeds but rl_eval_episodes={n_episodes}. "
                f"Delete the file to regenerate, or fix rl_eval_episodes to match."
            )
        return seeds
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_episodes).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f, indent=2)
    logging.info(f"[rigorous-eval] Generated {n_episodes} fixed episode seeds "
                 f"(master_seed={master_seed}): {seeds_path}")
    return seeds


def get_rigorous_eval_seeds(cfg, output_dir, n_episodes, master_seed):
    """The fixed seeds of EVERY rigorous evaluation of a run (rl_eval_episodes of them).

    rl_eval_seeds_file (optional YAML key): path of the episode_seeds.json written by the SFT evaluation
    (scripts/eval_curve.py, <output-dir>/episode_seeds.json). It is used as is, so the in-training evaluation runs on
    EXACTLY the episodes the SFT checkpoints were evaluated on. Without it, the seeds are generated like
    eval_curve.py does (np.random.default_rng(rl_eval_seed).integers(0, 2**31 - 1, size=rl_eval_episodes)), which gives
    the same list when the SFT evaluation used the same seed and number of episodes."""
    seeds_file = getattr(cfg, "rl_eval_seeds_file", None)
    if seeds_file:
        with open(seeds_file) as f:
            seeds = [int(x) for x in json.load(f)]
        if len(seeds) != n_episodes:
            raise ValueError(
                f"rl_eval_seeds_file {seeds_file} has {len(seeds)} seeds but rl_eval_episodes={n_episodes}: "
                f"set rl_eval_episodes to {len(seeds)} or point to another file."
            )
        logging.info(f"[rigorous-eval] {len(seeds)} fixed episode seeds read from {seeds_file}")
        return seeds
    return [int(x) for x in get_or_create_episode_seeds(output_dir, n_episodes, master_seed)]


def get_step0_seeds(output_dir, window, master_seed, eval_seeds=None):
    """Seeds of the step-0 baseline pass (success_rate_window episodes of the frozen base policy).

    Rigorous evaluation OFF (eval_seeds None): unchanged, get_or_create_episode_seeds(window).
    Rigorous evaluation ON: the FIRST len(eval_seeds) seeds ARE the evaluation seeds, so that the step-0 value of
    eval_rigorous/success_rate is measured on the same episodes as every later evaluation; the remaining ones (when the
    window is larger) come from np.random.default_rng(master_seed + 1), cached in step0_episode_seeds.json."""
    if not eval_seeds:
        return get_or_create_episode_seeds(output_dir, window, master_seed)
    n = len(eval_seeds)
    if window < n:
        raise ValueError(
            f"success_rate_window={window} is smaller than rl_eval_episodes={n}: the step-0 value of eval_rigorous is "
            f"taken from the step-0 pass, which must therefore cover at least rl_eval_episodes episodes."
        )
    if window == n:
        return list(eval_seeds)
    seeds = list(eval_seeds) + np.random.default_rng(master_seed + 1).integers(0, 2**31 - 1, size=window - n).tolist()
    path = Path(output_dir) / "step0_episode_seeds.json"
    if path.exists():
        with open(path) as f:
            cached = json.load(f)
        if cached != seeds:
            raise ValueError(f"{path} does not match the current evaluation seeds: delete it to regenerate.")
    else:
        with open(path, "w") as f:
            json.dump(seeds, f, indent=2)
    return seeds


def rigorous_eval_due(step, interval, last_bucket, at_episode_end, single_sim_env):
    """Is a rigorous evaluation due now?

    Environments with their own simulation per instance (ManiSkill): exactly at the multiples of `interval` (unchanged).
    Servers with ONE simulation shared by every connection (Isaac Lab / FORGE: the eval environment IS the training
    one): only between two training episodes, at the first episode end at or after each multiple of `interval`
    (`last_bucket` = step // interval of the last evaluation done, or of the start of the run)."""
    if interval <= 0 or step <= 0:
        return False
    if single_sim_env:
        return bool(at_episode_end) and step // interval > last_bucket
    return step % interval == 0


def append_rigorous_eval_record(log_path, record):
    """One JSON line per evaluation in <run>/checkpoints/eval_rigorous.jsonl: what was measured, on which seeds, with
    which result per episode. The file survives a wandb failure; it never raises."""
    try:
        with open(str(log_path), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logging.error(f"[rigorous-eval] Could not append to {log_path}: {e}")


def as_env_chunk(action_chunk, observation, cfg):
    """Chunk of the model -> chunk the environment executes (expo_ft/utils/action_space.py). `observation` is the one the
    chunk was planned from. Identity unless the task YAML sets chunk_action_reference."""
    return chunk_to_env_actions(
        action_chunk, observation, getattr(cfg, "chunk_action_reference", None), getattr(cfg, "state_obs_key", None)
    )


def run_rigorous_eval(agent, eval_env, episode_seeds, cfg, only_base_actions=False, episode_records=None):
    """Deterministic, fixed-seed evaluation -- Jesse's proposed protocol.
    Runs episode_seeds' episodes with agent.sample_actions(..., deterministic=True)
    (residual actor's mode, not a stochastic sample -- see expo_ft.py's
    sample_actions docstring) and returns (success_rate, stderr).

    only_base_actions: at step 0 (before any RL training), the residual and
    critic are freshly, randomly initialized -- deterministic=True only
    makes their OWN output deterministic, it does NOT mean "no residual
    contribution." Without only_base_actions=True, the step-0 number would
    actually measure SFT + an arbitrary untrained-residual offset + an
    untrained-critic argmax selection, not the clean frozen-SFT baseline
    Jesse's protocol calls for as the reference point. only_base_actions=True
    bypasses the whole residual/critic mechanism, matching exactly how
    eval_policy.py's own baseline mode works. Note this makes the eval
    genuinely non-deterministic at step 0 regardless of the deterministic=True
    passed to sample_actions internally -- the frozen base VLA's own
    sampling stays stochastic either way (no equivalent deterministic mode
    for a flow-matching model), which is expected and fine: there's no
    trained residual to speak of yet at step 0, so "deterministic" has
    nothing to apply to. Should be False for every later, trained-agent
    call (only_base_actions defaults to False, matching that).

    Deliberately does NOT touch the `agent` object passed in: uses its own
    local `eval_agent` variable, reassigned across this function's own
    rollout steps exactly like sample_actions() always requires (each call
    returns a new agent with an advanced .rng), but that variable is purely
    local and discarded when this function returns. The caller's own
    `agent` (and its .rng sequence) is completely unaffected by having run
    this evaluation -- training resumes exactly as if this detour never
    happened, only wandb gets the new eval_rigorous/* numbers.

    episode_records: optional list; when given, one {"seed", "success", "steps"} dict is appended per episode (what
    eval_rigorous.jsonl stores)."""
    from collections import deque
    eval_agent = agent
    successes = []
    for ep in tqdm.tqdm(range(len(episode_seeds)), desc="[rigorous-eval]", disable=not FLAGS.tqdm):
        obs = eval_env.reset(seed=int(episode_seeds[ep]))
        done = False
        steps = 0
        success = False
        action_plan = deque()
        while not done and steps < cfg.max_steps_per_episode:
            if not action_plan:
                action_chunk, eval_agent, _ = eval_agent.sample_actions(obs, only_base_actions=only_base_actions, deterministic=True)
                action_chunk = as_env_chunk(action_chunk, obs, cfg)
                action_plan.extend(action_chunk[:cfg.replan_steps])
            action = action_plan.popleft()
            eval_env.step(action.tolist())
            done, success, _, _ = eval_env.get_info_for_step()
            obs = eval_env.get_observation()
            steps += 1
        successes.append(float(success))
        if episode_records is not None:
            episode_records.append({"seed": int(episode_seeds[ep]), "success": bool(success), "steps": int(steps)})
    successes = np.asarray(successes)
    success_rate = float(successes.mean())
    stderr = float(successes.std(ddof=1) / np.sqrt(len(successes))) if len(successes) > 1 else 0.0
    return success_rate, stderr


def reconstruct_success_window(replay_buffer, success_rate_window):
    """On resume, when checkpoint_buffer=true has restored real past
    transitions (see restore_replay_buffer in expo_ft/data/replay_buffer.py
    -- files are named by step and reloaded via insert() in that same
    order, so the buffer's own index order IS true temporal order), rebuild
    eval/success_rate's rolling window directly from that restored history
    instead of leaving it empty. Without this, _success_window stays empty
    on every resume (it's a separate in-memory list, never populated by
    batch_processor.restore() itself) and eval/success_rate silently
    reports nothing until success_rate_window BRAND NEW episodes complete
    post-resume -- with a typical PushCube episode length, that's ~20K
    steps of pure silence on a metric checkpoint_buffer was specifically
    meant to make resume-safe.

    dones marks each episode's LAST transition; is_success at that same
    index is that episode's outcome (matches exactly how the live
    training loop appends to _success_window -- see the two call sites of
    training_log._success_window.append(...) in this file). Returns up to
    the last `success_rate_window` such outcomes, in original temporal
    order (oldest of the kept ones first) -- if fewer than
    success_rate_window episodes exist in the restored buffer, returns
    all of them (a partial pre-fill is still strictly better than none:
    it shrinks whatever real wait remains instead of eliminating it)."""
    size = replay_buffer._size
    if size == 0:
        return []
    dones = np.asarray(replay_buffer.dataset_dict["dones"][:size])
    is_success = np.asarray(replay_buffer.dataset_dict["is_success"][:size])
    episode_end_indices = np.flatnonzero(dones)
    if len(episode_end_indices) == 0:
        return []
    kept = episode_end_indices[-success_rate_window:]
    return is_success[kept].astype(float).tolist()


def _gpu_mem_report(tag, agent):
    """Diagnostic (only called when EXPO_FT_MEM_REPORT is set): what is on the GPU, and what part of it is the agent."""
    import collections
    try:
        live = jax.live_arrays()
        leaves = [x for x in jax.tree_util.tree_leaves(agent) if isinstance(x, jax.Array)]
        gib = 1024 ** 3
        agent_ids = {id(x) for x in leaves}
        uniq = {}
        for x in leaves:
            uniq[x.unsafe_buffer_pointer()] = x.nbytes
        others = [a for a in live if id(a) not in agent_ids]
        sizes = collections.Counter()
        counts = collections.Counter()
        for a in others:
            sizes[(tuple(a.shape), str(a.dtype))] += a.nbytes
            counts[(tuple(a.shape), str(a.dtype))] += 1
        logging.info("[memreport:%s] live arrays: %.2f GiB (%d) | agent leaves: %.2f GiB (%d) | agent distinct buffers: %.2f GiB (%d) | NOT in agent: %.2f GiB (%d)",
                     tag, sum(a.nbytes for a in live) / gib, len(live), sum(x.nbytes for x in leaves) / gib, len(leaves),
                     sum(uniq.values()) / gib, len(uniq), sum(a.nbytes for a in others) / gib, len(others))
        for (shape, dtype), nbytes in sizes.most_common(8):
            logging.info("[memreport:%s]   not in agent: %s %s x%d = %.2f GiB", tag, shape, dtype, counts[(shape, dtype)], nbytes / gib)
        # Real device memory: arrays can be several Python objects on ONE buffer (device_put to an equivalent sharding does
        # not copy), so count buffers, and say how much of what is outside the agent is NOT a view of an agent buffer.
        agent_ptrs = set(uniq)
        def _ptr(a):
            try:
                return a.unsafe_buffer_pointer()
            except Exception:
                return None
        all_ptrs = {}
        for a in live:
            q = _ptr(a)
            if q is not None:
                all_ptrs[q] = a.nbytes
        extra = {q: n for q, n in all_ptrs.items() if q not in agent_ptrs}
        logging.info("[memreport:%s] DISTINCT device buffers of all live arrays: %.2f GiB | of which not an agent buffer (real extra memory): %.2f GiB (%d buffers)",
                     tag, sum(all_ptrs.values()) / gib, sum(extra.values()) / gib, len(extra))
        by_shape = collections.Counter()
        cnt_shape = collections.Counter()
        for a in live:
            q = _ptr(a)
            if q in extra:
                by_shape[(tuple(a.shape), str(a.dtype))] += extra.pop(q)
                cnt_shape[(tuple(a.shape), str(a.dtype))] += 1
        for (shape, dtype), nbytes in by_shape.most_common(6):
            logging.info("[memreport:%s]   real extra: %s %s x%d = %.2f GiB", tag, shape, dtype, cnt_shape[(shape, dtype)], nbytes / gib)
        cache = getattr(agent, "_infer_cache", None) or {}
        for name, tree in cache.items():
            ls = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
            alias = sum(x.nbytes for x in ls if _ptr(x) in agent_ptrs)
            copy = sum(x.nbytes for x in ls if _ptr(x) not in agent_ptrs)
            logging.info("[memreport:%s] _infer_cache[%s]: %.2f GiB view of agent buffers, %.2f GiB real copy", tag, name, alias / gib, copy / gib)
        stats = jax.devices()[0].memory_stats()
        if stats:
            logging.info("[memreport:%s] device bytes_in_use %.2f GiB, peak %.2f GiB", tag,
                         stats.get("bytes_in_use", 0) / gib, stats.get("peak_bytes_in_use", 0) / gib)
    except Exception as e:  # diagnostic only
        logging.warning("[memreport:%s] failed: %s", tag, e)


def _reshard_restored_actor_state(agent):
    """After a checkpoint restore: put the actor's train state on the sharding a freshly built agent has.

    Orbax restores each array with the sharding read from the checkpoint's sharding file. If it is not equivalent to
    actor.train_state_sharding (the one the update programs are compiled for), the first update re-shards the weights into
    copies and cannot donate the originals ("Some donated buffers were not usable"): ~5 GiB more at its peak, which does not
    fit next to the Isaac Lab server on one L40S. device_put is a no-op for an array that is already well placed."""
    try:
        target = agent.actor.train_state_sharding
        leaves, treedef = jax.tree_util.tree_flatten(agent.actor_train_state)
        targets = jax.tree_util.tree_leaves(target, is_leaf=lambda x: isinstance(x, jax.sharding.Sharding))
        if len(leaves) != len(targets):
            logging.warning("[reshard] %d weights but %d target shardings: left as restored", len(leaves), len(targets))
            return agent
        bad = [i for i, (x, t) in enumerate(zip(leaves, targets))
               if isinstance(x, jax.Array) and not x.sharding.is_equivalent_to(t, x.ndim)]
        logging.info("[reshard] actor_train_state: %d of %d arrays (%.2f GiB) not on the sharding of a fresh agent",
                     len(bad), len(leaves), sum(leaves[i].nbytes for i in bad) / 1024 ** 3)
        if bad:
            for i in bad:
                try:
                    leaves[i] = jax.device_put(leaves[i], targets[i], donate=True)
                except TypeError:
                    leaves[i] = jax.device_put(leaves[i], targets[i])
            agent = agent.replace(actor_train_state=jax.tree_util.tree_unflatten(treedef, leaves))
    except Exception as e:
        logging.warning("[reshard] skipped (%s)", e)
    return agent


def main(_):
    init_logging()

    # Load task config from YAML
    cfg = load_task_config(FLAGS.task_config)
    # Task YAML -> model config (FLAGS.config), all in one place: openpi config name, the RL hyperparameters of the
    # learner (rl_* for EXPOLearner, ppo_* for PPOLearner, ... so one YAML can hold the tuned values of every algorithm
    # without collisions), actor_success_only, and the tactile / force input (use_tactile, use_force, force_mode).
    # Shared with scripts/eval_policy.py and the diagnostic scripts: expo_ft/utils/config_loader.py.
    aux_spec = apply_task_overrides(FLAGS.config, cfg)
    logging.info("Tactile / force input of the residual actor and the critic: %s", aux_spec.describe())

    # Read once, early — used for the learner-import dispatch further down.
    model_cls = FLAGS.config.model_cls

    # Action frame (chunk_action_reference, validated by apply_task_overrides): with "chunk_start_state" the model's chunk
    # is made of offsets from the joint state at plan time, and every chunk is turned into absolute joint targets right
    # after agent.sample_actions (as_env_chunk below); the replay buffer converts back, row by row, when it stores
    # the executed actions. None = one shared frame (ManiSkill), as_env_chunk is then the identity.
    logging.info("Action frame: chunk_action_reference=%s (state_obs_key=%s)",
                 FLAGS.config.chunk_action_reference, FLAGS.config.state_obs_key)

    # AssetsConfig DROID officielle deja bakee dans la config openpi nommee ci-dessus —
    # ne pas l'ecraser ici (meme bug corrige dans eval_policy.py : le SFT a ete
    # entraine avec ces stats officielles, pas des stats locales par repo_id).
    # Mark the run directory as an RL run (e.g. stack_cube_expo_ft_2026-07-05_01-06-12_rl)
    # so it's visually distinguishable from an SFT run directory at a glance.
    run_dir, resuming = resolve_run_dir(cfg, resume_dir=cfg.rl_resume_dir, suffix="rl")
    # Normalization statistics of the SFT checkpoint (<checkpoint>/assets/<id>/norm_stats.json, e.g. the ones computed
    # on the demos): used by the actor, the replay buffers and the evaluation of this run alike, and copied to
    # <run_dir>/assets/ so that the eval of its checkpoints and a resume find them. No-op for a DROID-stats SFT.
    from expo_ft.utils.train_utils import setup_rl_norm_stats
    setup_rl_norm_stats(FLAGS.config, run_dir)

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
        # Demonstrations of the task YAML, in the format it points to (demos_dir: FORGE .npz, droid_format_dir:
        # DROID format). offline_ratio and rl_seed_demos_online decide where they go (BatchProcessor), unchanged.
        dataset = load_demo_dataset(
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

    # Separate environment instance for periodic rigorous (deterministic,
    # fixed-seed) evaluation -- see run_rigorous_eval() below. Kept
    # completely separate from the training env so a mid-episode training
    # rollout is never disturbed by an eval detour. Only created when the
    # feature is actually enabled (rl_eval_interval > 0), matching how
    # rl_critic_pretrain_steps/rl_actor_pretrain_steps etc. are opt-in.
    rl_eval_interval = int(getattr(cfg, "rl_eval_interval", 0) or 0)
    rl_eval_episodes = int(getattr(cfg, "rl_eval_episodes", 200))
    rl_eval_seed = int(getattr(cfg, "rl_eval_seed", 42))
    # critic-only variant of the evaluation (n_edit_samples=0): ON when the key is absent (what every earlier run did),
    # the FORGE task YAMLs set it to false.
    rl_eval_critic_only = bool(getattr(cfg, "rl_eval_critic_only", True))
    # Isaac Lab / FORGE: the server hands out ONE simulation, so the "eval environment" is the training simulation
    # itself: the evaluation can only run between two training episodes (rigorous_eval_due).
    single_sim_env = getattr(cfg, "env_wrapper", None) in ("isaaclab", "remote")
    rl_eval_log_path = checkpoint_dir_path / "eval_rigorous.jsonl"
    eval_seeds = None
    eval_env = None
    if rl_eval_interval > 0:
        if rl_eval_episodes < 1:
            raise ValueError(f"rl_eval_episodes must be >= 1 when rl_eval_interval > 0, got {rl_eval_episodes}")
        # Read (or generated) NOW: a wrong rl_eval_seeds_file / rl_eval_episodes stops the run at the start, not hours in.
        eval_seeds = get_rigorous_eval_seeds(cfg, checkpoint_dir_path, rl_eval_episodes, rl_eval_seed)
        eval_env_creation_request = {
            "example_action": example_action,
            "env_usage": "eval",
            "video_dir": None,
        }
        if cfg.env_wrapper == "droid":
            eval_env = EnvClientWrapper(
                env_creation_request=eval_env_creation_request,
                host="localhost",
                port=8102,
            )
        else:
            eval_env = make_env_wrapper(env_creation_request=eval_env_creation_request, cfg=cfg)
        eval_env.reset()
        logging.info(f"Created separate rigorous-eval environment {eval_env.env_id} "
                     f"(every {rl_eval_interval} steps, {rl_eval_episodes} episodes, "
                     f"deterministic actor, fixed seeds from master_seed={rl_eval_seed})")
        logging.info(f"[rigorous-eval] {len(eval_seeds)} fixed seeds, critic-only variant "
                     f"{'ON' if rl_eval_critic_only else 'OFF'}, results appended to {rl_eval_log_path}. "
                     + ("Single-simulation server: evaluated between two training episodes (first episode end at or "
                        "after each multiple of rl_eval_interval); the training environment is reset right after."
                        if single_sim_env else "Separate environment instance: evaluated at every multiple of rl_eval_interval."))

    # model_cls already read near the top of main() (see hyperparameter
    # override block above) — reused here for the learner-import dispatch.
    # BCLearner uses human-intervention chunks for the actor batch only (no critic).
    use_dagger_hil_sampling = model_cls == "BCLearner"
    if model_cls == "BCLearner":
        from expo_ft.agents.alg.bc import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearner":
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "EXPOLearnerCategorical":
        # The categorical/distributional (XQC/XQCfD-style, C51-bounded
        # support) critic rewrite, kept in expo_ft_categorical.py — for
        # direct A/B comparison against the MSE scalar critic/REDQ-ensemble
        # architecture now used by the default "EXPOLearner". Shares the
        # exact same config overrides above (expo_ft_categorical.create()'s
        # signature ends in **kwargs, so the MSE/REDQ-specific fields set
        # there — num_qs, num_min_qs, critic_layer_norm, etc. — are silently
        # absorbed and ignored, not an error).
        from expo_ft.agents.alg.expo_ft_categorical import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "PPOLearner":
        from expo_ft.agents.alg.ppo import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "GRPOLearner":
        from expo_ft.agents.alg.grpo import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "PPOResidualLearner":
        # Same on-policy PPO machinery as PPOLearner, but trained on a
        # residual policy conditioned on the frozen VLA's own sampled base
        # action (via ExpoFT's own residual actor construction), instead of
        # a standalone actor with zero connection to it — see
        # ppo_residual.py's module docstring for the full rationale.
        from expo_ft.agents.alg.ppo_residual import load_agent, restore_checkpoint, save_checkpoint
    elif model_cls == "GRPOResidualLearner":
        # GRPO counterpart of PPOResidualLearner — same residual mechanism,
        # group-relative advantage instead of GAE. See grpo_residual.py.
        from expo_ft.agents.alg.grpo_residual import load_agent, restore_checkpoint, save_checkpoint
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
    # Force zero demo contamination for these four model classes (the plain
    # and residual variants alike), regardless of whatever offline_ratio
    # happens to be set to in the task YAML.
    is_on_policy_algo = model_cls in ("PPOLearner", "GRPOLearner", "PPOResidualLearner", "GRPOResidualLearner")
    # Residual variants additionally need the frozen VLA's own sampled base
    # action recorded per-transition (see ppo_residual.py/grpo_residual.py's
    # module docstrings): update() must recompute the residual's log_prob
    # against the EXACT base action used at rollout time, not a fresh
    # resample (pi0.5 is a stochastic flow-matching model, so a fresh sample
    # would generally differ from the one actually executed).
    is_residual_on_policy_algo = model_cls in ("PPOResidualLearner", "GRPOResidualLearner")
    # Number of consecutive transitions per on-policy rollout. Fixed (rather
    # than "however many the last episode happened to last") so the batch
    # shape stays constant across updates — a varying shape would force JAX to
    # recompile _update_jit on every single update. Unused by off-policy
    # learners.
    rollout_length = int(getattr(FLAGS.config, "rollout_length", 0) or 0)
    if is_on_policy_algo:
        if rollout_length <= 0:
            raise ValueError(
                f"model_cls={model_cls} is on-policy and requires rollout_length > 0 "
                f"(set ppo_rollout_length / grpo_rollout_length in the task YAML)"
            )
        _nmb = int(getattr(FLAGS.config, "num_minibatches", 1) or 1)
        if rollout_length % _nmb != 0:
            # _update_jit asserts this too, but only once inside the traced
            # update — failing here instead surfaces it before a long rollout
            # collection phase has already been spent.
            raise ValueError(
                f"rollout_length ({rollout_length}) must be divisible by "
                f"num_minibatches ({_nmb})"
            )
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
        store_base_actions=is_residual_on_policy_algo,
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
        # Independent of offline_ratio's value -- see BatchProcessor's own
        # docstring/comment. Off (default) means genuinely no demos anywhere
        # unless offline_ratio > 0 also puts them in the offline buffer;
        # explicitly opt in via rl_seed_demos_online in the task YAML to
        # match the EXPO paper's own single-buffer convention instead.
        seed_demos_online=False if is_on_policy_algo else bool(getattr(cfg, "rl_seed_demos_online", False)),
        # PPO/GRPO need a contiguous, time-ordered rollout collected under the
        # current policy, not a uniform random draw over the whole buffer
        # history — see BatchProcessor's docstring. Off-policy learners keep
        # the previous behavior exactly (on_policy=False).
        on_policy=is_on_policy_algo,
        rollout_length=rollout_length if is_on_policy_algo else 0,
        replan_steps=cfg.replan_steps,
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
        if os.environ.get("EXPO_FT_MEM_REPORT"):
            _gpu_mem_report("before_restore", agent)
        agent = restore_checkpoint(checkpoint_manager, agent)
        agent = _reshard_restored_actor_state(agent)
        agent = agent.cache_infer_params()
        # The restored agent has its own arrays. These three locals (only used by load_agent above) still point to the
        # arrays the agent had BEFORE the restore (SFT weights): on a fresh run they are the agent's own, on a resume they
        # would stay on the GPU as a useless second copy for the whole run.
        actor_train_state = target_actor_params = agent_kwargs = None
        gc.collect()
        if os.environ.get("EXPO_FT_MEM_REPORT"):
            _gpu_mem_report("after_restore", agent)
            logging.info("[memreport] EXPO_FT_MEM_REPORT is set: report done, stopping before the training starts.")
            raise SystemExit(0)
        steps = tuple(checkpoint_manager.all_steps())
        latest_step = max(steps) if steps else None
        if latest_step is not None:
            start_step = latest_step
            logging.info("Resuming from step %d", start_step)
        batch_processor.restore(checkpoint_dir_path, up_to_step=latest_step)

    # Computed here (right after restore, while the restored buffer is
    # freshest) but only actually USED later at training_log._success_window's
    # own initialization, since training_log doesn't exist yet at this point.
    # See reconstruct_success_window()'s own docstring for why this exists.
    _restored_success_window = (
        reconstruct_success_window(batch_processor.replay_buffer, int(getattr(cfg, "success_rate_window", 200)))
        if resuming and cfg.checkpoint_buffer else []
    )

    # Demos live in offline_replay_buffer when offline_ratio > 0 (the normal
    # ExpoFT setup); BatchProcessor's constructor instead seeds them into the
    # online replay_buffer when offline_ratio == 0 — follow whichever buffer
    # actually received the dataset. Shared by both pretraining stages below.
    # Guarded by model_cls: offline_ratio/this whole pretraining mechanism is
    # ExpoFT-specific (EXPOLearner/EXPOLearnerCategorical) -- PPO/GRPO/SAC task YAMLs
    # don't define offline_ratio at all, so this must not run unconditionally.
    critic_pretrain_steps = int(getattr(FLAGS.config, "critic_pretrain_steps", 0) or 0)
    actor_bc_pretrain_steps = int(getattr(FLAGS.config, "actor_bc_pretrain_steps", 0) or 0)
    if model_cls in ("EXPOLearner", "EXPOLearnerCategorical"):
        pretrain_buffer = offline_replay_buffer if cfg.offline_ratio > 0 else replay_buffer
    else:
        pretrain_buffer = None

    # ── Discarded compile-only warm-up, resumed runs only ───────────────────
    # Why: agent.update()'s first-ever call (and thus first-ever JIT
    # compilation of the whole update pipeline) is otherwise gated behind
    # `training_log.ep_count >= 10` in the main loop below, which resets to 0
    # on every resume regardless of prior training -- see that check. On a
    # resumed run this delays first compilation ~700+ steps in, by which
    # point whatever compile-time peak memory that first compilation needs
    # has to coexist with everything accumulated in the meantime, on top of
    # the checkpoint restore's own ~18GB. This repeatedly produced
    # RESOURCE_EXHAUSTED crashes at that exact point (never on a fresh run,
    # which never carries the extra restore memory).
    #
    # Triggering that same compilation HERE instead -- immediately after
    # restore, before rollout collection has a chance to add anything else
    # to the picture -- is a genuine attempt at reducing what's
    # simultaneously resident at the moment of peak demand, not just
    # relocating the same peak. Both static branches of _update_finalize_jit
    # (use_success_batch True and False) are warmed deliberately, so neither
    # is deferred to whenever a real success episode first appears online.
    #
    # Deliberately does NOT depend on pretrain_buffer/demo data being
    # available anywhere -- robust to offline_ratio=0.0 + rl_seed_demos_online
    # =false (the genuinely-no-demos-anywhere config), where pretrain_buffer
    # is correctly empty by design, not by bug. Instead this samples directly
    # from `replay_buffer` (the online buffer, which always exists, is always
    # pre-allocated to capacity=cfg.max_steps >> cfg.batch_size, and is safe
    # to read at literal indices [0, batch_size) via sample_jax(indices=...),
    # which bypasses the usual "only sample within _size" restriction).
    #
    # Those specific indices are explicitly zeroed first for the CONTROL
    # fields (dones/masks/is_hil/hil_chunk/rewards) before sampling --
    # PiReplayBuffer.__init__ allocates these via np.empty, not np.zeros
    # (only is_success is zero-initialized), so reading them unwritten would
    # be uninitialized garbage, which risks corrupting _finalize_sample's
    # "next" index bookkeeping (n-step reward packing) rather than just
    # producing meaningless-but-safe numbers the way garbage image/state/
    # action data would. Image/state/action fields are left as whatever
    # np.empty gave them -- irrelevant here since the result is discarded
    # and no control-flow logic depends on their values.
    #
    # This never touches _size/_insert_index, so it can never be sampled by
    # real training later (normal sampling only draws from [0, _size), which
    # stays 0 here) and real inserts will naturally overwrite these same
    # slots first anyway (_insert_index starts at 0) -- zero contamination
    # risk to the real training trajectory.
    #
    # Wrapped in try/except: this is a best-effort optimization, not a
    # required step. If it fails for any reason, the run falls back to the
    # original behavior (compilation deferred to the ep_count gate as
    # before) rather than failing a run that might not even have hit the
    # crash this exists for.
    _warmup_agent = _discard_agent = _discard_agent2 = _warmup_batch = _actor_warmup_batch = None
    if resuming and getattr(cfg, "resume_warmup", True):
        try:
            logging.info("[warmup] Triggering discarded warm-up compilation of agent.update() "
                         "right after restore (both use_success_batch branches), before rollout "
                         "collection accumulates further memory pressure...")
            # agent.update() internally splits its input batch into utd_ratio
            # minibatches of cfg.batch_size each (see _prepare_minibatches_jit's
            # own assert total_bs % utd_ratio == 0) -- so the TOTAL batch this
            # call needs is cfg.batch_size * cfg.utd_ratio, matching exactly
            # how BatchProcessor.__init__ computes its own total_bs for the
            # real training iterators. Passing cfg.batch_size alone here
            # (24, not divisible by utd_ratio=20) is what the previous
            # attempt got wrong.
            _total_bs = cfg.batch_size * cfg.utd_ratio
            _warmup_n = _total_bs + cfg.replan_steps + 1  # safety margin for "next" index lookups
            for _field in ("dones", "masks", "is_hil", "hil_chunk", "rewards"):
                if _field in replay_buffer.dataset_dict:
                    replay_buffer.dataset_dict[_field][:_warmup_n] = 0
            # sample_jax() asserts len(self) >= self._replan_steps UNCONDITIONALLY,
            # before it even looks at `indices` -- with _size genuinely 0 (nothing
            # inserted yet on a resumed run), that assertion fails regardless of
            # indices being explicitly given. Temporarily report a large-enough
            # size just for this call, then restore the true value immediately
            # after -- consistent with directly poking dataset_dict above; _size
            # is what real training's own insert()/sample_jax() calls rely on
            # being accurate, so it must not be left altered.
            _true_size = replay_buffer._size
            replay_buffer._size = max(_true_size, _warmup_n)
            try:
                _warmup_indices = np.arange(_total_bs)
                # sample_jax's raw output still uses the buffer's own field
                # names (base_image, left_wrist_image, ...) -- agent.update()
                # expects the assembled "image"/"next_image" dict structure
                # instead. _convert_to_openpi_format does exactly this
                # conversion; apply_data_sharding is applied AFTER it (not
                # via sample_jax's own data_sharding= param), matching
                # next_batch()'s own proven on-policy-branch ordering exactly
                # rather than inventing a different one here.
                _warmup_raw = replay_buffer.sample_jax(_total_bs, indices=_warmup_indices)
                _warmup_batch = replay_buffer._convert_to_openpi_format(_warmup_raw)
                _warmup_batch = replay_buffer.apply_data_sharding(_warmup_batch, data_sharding)
            finally:
                replay_buffer._size = _true_size

            # agent's buffers may be donated inside update()'s own internal
            # JIT calls (see expo_ft.py's donate_argnames=("agent",) on
            # _prepare_minibatches_jit/_critic_update_step_jit/
            # _update_finalize_jit) -- reusing the SAME agent reference for
            # a second call after donation would be unsafe (its buffers may
            # already be reused/invalid). Chaining instead: the second warm-up
            # call uses the first call's own (still-discarded) result as its
            # input, so the original `agent` reference is only ever used
            # once, by the first call.
            # CRITICAL: donation must never touch the REAL, persistent
            # `agent` object -- it's reused for the rest of training
            # (sample_actions, etc.) right after this block. The earlier
            # "chain the second call through the first's result" fix only
            # protected the SECOND warmup call; it missed that the FIRST
            # call itself passes the real `agent` as the donated argument,
            # letting JAX free ITS buffers too. That's exactly what caused
            # "RuntimeError: Array has been deleted with shape=uint32[2]"
            # later in sample_actions -- agent.rng's buffer had already
            # been donated away during warmup. A genuine, independent
            # .copy() of every array leaf here ensures donation only ever
            # touches a throwaway copy, never the real agent.
            _warmup_agent = jax.tree_util.tree_map(
                lambda x: x.copy() if isinstance(x, jax.Array) else x, agent
            )
            _discard_agent, _discard_info = _warmup_agent.update(_warmup_agent, _warmup_batch, cfg.utd_ratio, None)
            jax.block_until_ready(_discard_agent)
            del _discard_info, _warmup_agent

            # Second pass warms the use_success_batch=True branch too (see
            # comment block above). That branch's actor_batch goes through
            # its OWN separate prepare_critic_batch call (see
            # _update_finalize_jit), NOT the utd_ratio-based minibatch
            # splitting _prepare_minibatches_jit does -- so it expects
            # cfg.batch_size directly, not total_bs. Sampled separately here
            # rather than reusing _warmup_batch, which is the wrong size for
            # this specific argument.
            _actor_warmup_indices = np.arange(cfg.batch_size)
            replay_buffer._size = max(_true_size, cfg.batch_size + cfg.replan_steps + 1)
            try:
                _actor_warmup_raw = replay_buffer.sample_jax(cfg.batch_size, indices=_actor_warmup_indices)
                _actor_warmup_batch = replay_buffer._convert_to_openpi_format(_actor_warmup_raw)
                _actor_warmup_batch = replay_buffer.apply_data_sharding(_actor_warmup_batch, data_sharding)
            finally:
                replay_buffer._size = _true_size
            # _discard_agent is already a throwaway copy's descendant (never
            # shared any buffers with the real, persistent agent to begin
            # with), so chaining through it here remains safe.
            _discard_agent2, _discard_info2 = _discard_agent.update(_discard_agent, _warmup_batch, cfg.utd_ratio, _actor_warmup_batch)
            jax.block_until_ready(_discard_agent2)
            del _discard_agent, _discard_agent2, _discard_info2, _actor_warmup_batch

            del _warmup_batch
            gc.collect()
            logging.info("[warmup] Compile-only warm-up complete; agent unchanged "
                         "(warm-up result discarded, never assigned; only the "
                         "original `agent` reference passed to update() -- the "
                         "chained second call never reused it after donation).")
        except Exception as e:
            logging.warning("[warmup] Warm-up compilation attempt failed (%s) -- continuing "
                           "with the original deferred-compilation behavior instead.", e)
            # The copy of the agent made for the warm-up (and its descendants) stays referenced by these locals when the
            # warm-up fails halfway: release it, otherwise it stays on the GPU for the whole run.
            _warmup_agent = _discard_agent = _discard_agent2 = _warmup_batch = _actor_warmup_batch = None
            gc.collect()
    elif resuming:
        logging.info("[warmup] resume_warmup=false: no warm-up compilation, agent.update() compiles at its first real call.")

    # ── Actor BC warm-start (XQCfD-style "policy pretraining") ──────────────
    # Only on a fresh run, same rationale as critic pretraining below. Runs
    # BEFORE critic pretraining — see pretrain_actor_bc()'s docstring for why
    # the ordering matters. EXPOLearner (MSE) doesn't have pretrain_actor_bc
    # (legacy baseline, frozen on purpose) — only EXPOLearnerCategorical
    # (the categorical-critic architecture) does.
    if not resuming and model_cls == "EXPOLearnerCategorical" and actor_bc_pretrain_steps > 0:
        logging.info(
            "BC warm-start: pretraining residual actor for %d steps on demo data (%s buffer)...",
            actor_bc_pretrain_steps,
            "offline" if cfg.offline_ratio > 0 else "online (offline_ratio=0, demos live there instead)",
        )
        bc_iterator = pretrain_buffer.get_iterator(
            sample_args={"batch_size": cfg.batch_size},
            data_sharding=data_sharding,
        )
        wandb.define_metric("bc_pretrain_step")
        wandb.define_metric("bc_pretrain/*", step_metric="bc_pretrain_step")
        for bc_step in tqdm.tqdm(
            range(actor_bc_pretrain_steps), desc="Actor BC warm-start", disable=not FLAGS.tqdm
        ):
            bc_batch = next(bc_iterator)
            bc_batch = pretrain_buffer.apply_data_sharding(bc_batch, data_sharding)
            bc_batch = dict(bc_batch)
            rng, key1 = jax.random.split(agent.rng)
            bc_batch["image"] = agent.data_augmentation_fn(key1, bc_batch["image"])
            # next_image left unaugmented on purpose: pretrain_actor_bc never
            # reads next_observations, but prepare_critic_batch still builds
            # that key from whatever's in next_image — passing it through raw
            # is harmless and saves an augmentation call.
            bc_batch = prepare_critic_batch(
                bc_batch,
                agent.actor.model_config.action_dim,
                agent.action_dim,
                agent.state_dim,
                agent.action_horizon,
                agent.replan_steps,
            )
            agent = agent.replace(rng=jax.device_put(rng, replicated_sharding))
            agent, bc_info = agent.pretrain_actor_bc(bc_batch)
            for k, v in bc_info.items():
                try:
                    tb_writer.add_scalar(f"bc_pretrain/{k}", float(v), global_step=bc_step)
                except (TypeError, ValueError):
                    pass
            wandb.log({
                "bc_pretrain_step": bc_step,
                **{f"bc_pretrain/{k}": v for k, v in bc_info.items()},
            })
        logging.info("Actor BC warm-start complete.")

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
    if not resuming and model_cls in ("EXPOLearner", "EXPOLearnerCategorical") and critic_pretrain_steps > 0:
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

    # =====================================================================
    # Priming phase: evaluate the STARTING model on success_rate_window real
    # episodes, BEFORE any weight updates. Deliberately kept OUTSIDE
    # cfg.max_steps' budget -- has its own step counter here, but `i` and
    # everything keyed off it below (checkpoint naming, cfg.max_steps, the
    # loop bound itself) are completely untouched by this phase.
    #
    # wandb rejects going backward on its own internal step counter -- but
    # ALSO, tested directly (offline run, raw datastore inspection), NEVER
    # passing an explicit step= at all and instead using per-metric custom
    # step metrics (wandb.define_metric(..., step_metric=...)) sidesteps
    # that entirely: the internal counter still auto-increments harmlessly
    # in the background (nothing reads it), while each metric's own chart
    # x-axis uses whatever value we log alongside it under its custom step
    # key. This means training/eval metrics can genuinely start at 0 for
    # every run/task -- clean, directly comparable curves, no offset
    # arithmetic needed -- while eval/init_progress (priming) keeps its own
    # separate, real-step-based x-axis. Checkpoint saving/naming and
    # cfg.max_steps below use `i` directly, unaffected either way.
    #
    # Skipped entirely when resuming: the model already has real training
    # history; _success_window is a plain in-memory list (not part of the
    # checkpoint) so it isn't "empty" in any meaningful sense on resume,
    # there's just nothing to reconstruct it from -- re-priming would
    # needlessly redo real training time.
    wandb.define_metric("eval/init_step")
    wandb.define_metric("eval/init_progress", step_metric="eval/init_step")
    wandb.define_metric("training/global_step")
    wandb.define_metric("training/*", step_metric="training/global_step")
    wandb.define_metric("eval/success_rate", step_metric="training/global_step")
    # Deterministic, fixed-seed rigorous eval (Jesse's proposed protocol) --
    # deliberately a DIFFERENT metric name from eval/success_rate above
    # (that one is the in-training rolling window over stochastic online
    # episodes) so the two are never confused on the same dashboard.
    wandb.define_metric("eval_rigorous/success_rate", step_metric="training/global_step")
    wandb.define_metric("eval_rigorous/success_rate_stderr", step_metric="training/global_step")
    wandb.define_metric("eval_rigorous/n_episodes", step_metric="training/global_step")
    wandb.define_metric("eval_rigorous/seconds", step_metric="training/global_step")
    # Same protocol, but with n_edit_samples=0: the trained critic still
    # picks (argmax) among the N raw/un-edited VLA candidate samples, but the
    # residual actor is never invoked. Isolates how much of eval_rigorous's
    # improvement over the frozen-SFT baseline comes from the critic's
    # candidate selection alone, vs. the residual correction on top of it.
    wandb.define_metric("eval_rigorous/success_rate_critic_only", step_metric="training/global_step")
    wandb.define_metric("eval_rigorous/success_rate_critic_only_stderr", step_metric="training/global_step")

    success_rate_window = getattr(cfg, "success_rate_window", 200)
    training_log._success_window = _restored_success_window
    if _restored_success_window:
        logging.info(f"[resume] Reconstructed eval/success_rate's rolling window from "
                     f"{len(_restored_success_window)} restored episode(s) in the checkpointed "
                     f"replay buffer ({'full' if len(_restored_success_window) >= success_rate_window else 'partial'} "
                     f"window of {success_rate_window}) -- no need to wait for that many brand-new "
                     f"episodes post-resume before this metric reports again.")

    # ── Unified step-0 baseline pass (fresh runs only) ──────────────────────
    # One set of success_rate_window fixed-seed episodes, run on the TRAINING
    # env with only_base_actions=True (the true, clean frozen-SFT baseline --
    # no trained residual/critic exists yet at step 0, so this is the only
    # way to avoid biasing every later "did training improve over baseline"
    # comparison with an arbitrary untrained-residual contribution). This one
    # pass now serves three purposes at once, instead of two separate
    # 200-episode passes as before: (1) seeds the replay buffer with real
    # transitions via batch_processor.insert_transition, exactly as the old
    # priming phase did: (2) fills training_log._success_window, giving
    # eval/success_rate its own step-0 value; (3) gives eval_rigorous/
    # success_rate its step-0 value too, computed from the SAME episode
    # outcomes.
    #
    # Robust to eval_env being None (rl_eval_interval=0, rigorous eval
    # disabled entirely): this whole pass, and eval/success_rate's own
    # initialization, do NOT depend on eval_env or rl_eval_interval at
    # all -- only whether eval_rigorous/* also gets computed and logged
    # from the same data is conditional on that.
    #
    # Uses FIXED episode seeds (get_or_create_episode_seeds, the same
    # mechanism run_rigorous_eval's periodic calls use) rather than the
    # env's own natural reset randomness -- makes this step-0 pass
    # reproducible across different fresh runs of the same task (e.g.
    # comparing hyperparameter configs, all starting from the identical
    # 200 conditions), and is what makes deriving eval_rigorous/success_rate
    # from this same pass valid at all.
    if not resuming and success_rate_window > 0:
        step0_seeds = get_step0_seeds(checkpoint_dir_path, success_rate_window, rl_eval_seed, eval_seeds)

        # Large fixed offset for this pass's own saved-transition step
        # numbers: guaranteed far beyond any realistic cfg.max_steps (real
        # training steps top out in the 1e5-1e6 range for these tasks), so
        # step-0 saves can never collide with real training-step saves in
        # the same buffers/ directory. Must stay non-negative: restore_replay_buffer
        # filters filenames with str.isdigit(), which is False for a
        # leading "-", so a negative offset would be silently excluded on
        # restore -- this bit us once already while designing this.
        STEP0_STEP_OFFSET = 10**9

        # Detect a step-0 pass that was interrupted by preemption: no agent
        # checkpoint exists yet (real training never started, hence
        # `resuming=False` above), but if checkpoint_buffer=true, some of
        # THIS pass's own transitions may already be saved to disk from a
        # previous attempt. Restore them and skip the seeds already
        # covered, instead of redoing the whole pass from scratch -- this
        # phase never updates the agent's own weights ("No agent.update(...)
        # call anywhere in this phase, by design", below), so there is
        # nothing to restore about the agent itself: the SFT checkpoint
        # freshly loaded via pi05_weight_loader_path already IS the correct,
        # unchanged starting point either way. Only the already-collected
        # transitions and episode outcomes are worth recovering.
        step0_already_done = 0
        step0_transitions_restored = 0
        if cfg.checkpoint_buffer:
            try:
                batch_processor.restore(checkpoint_dir_path)
                step0_transitions_restored = batch_processor.replay_buffer._size
                _restored_step0_window = reconstruct_success_window(batch_processor.replay_buffer, success_rate_window)
                if _restored_step0_window:
                    training_log._success_window = _restored_step0_window
                    step0_already_done = len(_restored_step0_window)
                    logging.info(f"[step-0] Found {step0_already_done} already-completed episode(s) "
                                 f"({step0_transitions_restored} raw transitions) saved from an interrupted "
                                 f"previous attempt at this same pass -- resuming from episode "
                                 f"{step0_already_done + 1}/{success_rate_window} instead of restarting "
                                 f"from scratch.")
            except Exception as e:
                logging.warning(f"[step-0] Could not check for an interrupted previous attempt "
                                 f"({e}) -- starting this pass from episode 1 as normal.")

        logging.info(f"[step-0] Evaluating the frozen SFT base policy (only_base_actions=True) "
                     f"on {success_rate_window} fixed-seed episodes -- seeds the replay buffer, "
                     f"initializes eval/success_rate's rolling window, and (if rl_eval_interval>0) "
                     f"gives eval_rigorous/success_rate its own step-0 value, all from this one pass...")
        # Resume the RAW-TRANSITION count from wherever the restored buffer
        # left off (0 on a genuinely fresh attempt) -- this is a small,
        # human/wandb-readable number (at most a few thousand for a
        # realistic success_rate_window), completely SEPARATE from
        # step0_file_step below. Keeping these two counters distinct is the
        # important part: priming_step is what gets logged to wandb
        # (eval/init_step, and indirectly eval/init_progress's own x-axis,
        # since it's that metric's declared step_metric) and must stay
        # small; step0_file_step is only ever used for on-disk filenames and
        # is the one that needs STEP0_STEP_OFFSET, to guarantee no collision
        # with the main training loop's own step-numbered files. An earlier
        # version of this fix used ONE variable for both purposes, which
        # correctly fixed the file-naming collision risk but also
        # accidentally sent eval/init_step (and eval/init_progress's x-axis)
        # up into the billions -- this split is the actual fix.
        priming_step = step0_transitions_restored
        step0_file_step = STEP0_STEP_OFFSET + step0_transitions_restored
        step0_episode_idx = step0_already_done
        priming_pbar = tqdm.tqdm(total=success_rate_window, initial=step0_already_done, desc="step-0 eval", disable=not FLAGS.tqdm)
        env.reset(seed=int(step0_seeds[step0_episode_idx]))

        while len(training_log._success_window) < success_rate_window:
            observation = env.get_observation()

            if not action_plan and action_type != "human":
                # only_base_actions=True: see this block's own comment above
                # for why -- the clean frozen-SFT baseline, not SFT plus an
                # arbitrary untrained-residual perturbation.
                action_chunk, agent, new_si = agent.sample_actions(observation, only_base_actions=True)
                action_chunk = as_env_chunk(action_chunk, observation, cfg)
                episode_log.sample_info_history.append(new_si)
                action_plan.extend(action_chunk[:cfg.replan_steps])
            else:
                episode_log.sample_info_history.append(
                    episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                )

            elapsed = time.time() - start_step_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

            has_action = bool(action_plan)
            action = action_plan.popleft() if has_action else np.zeros_like(example_action.squeeze())
            real_action, action_type = env.step(action.tolist())
            start_step_time = time.time()
            done, success, reward, mask = env.get_info_for_step()
            priming_step += 1
            step0_file_step += 1

            episode_log.record_step(observation, len(action_plan), action_type, real_action, reward)

            if action_type == "human":
                action_plan.clear()

            if has_action or action_type == "human":
                _last_si = episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
                transition_dict = dict(
                    observations=observation, actions=real_action, rewards=reward,
                    masks=mask, dones=done, is_hil=(action_type == "human"),
                )
                if _last_si and "base_action" in _last_si:
                    transition_dict["base_actions"] = _last_si["base_action"]
                batch_processor.insert_transition(transition_dict)
                # Persist this pass's own transitions to disk too (offset
                # step numbers, see STEP0_STEP_OFFSET above) -- without
                # this, a preemption mid-pass had no way to resume short of
                # redoing the whole 200 episodes, even with
                # checkpoint_buffer=true, since that flag previously only
                # covered the MAIN training loop's own saves.
                if cfg.checkpoint_buffer:
                    try:
                        save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=step0_file_step)
                    except Exception as e:
                        logging.error(f"[step-0] Could not save replay buffer transition: {e}")
            # No agent.update(...) call anywhere in this phase, by design.

            if done:
                batch_processor.on_episode_done(success)
                training_log.on_episode_done(episode_log, success, {})
                training_log._success_window.append(float(success))
                priming_pbar.update(1)
                wandb.log({
                    "eval/init_progress": len(training_log._success_window) / success_rate_window,
                    "eval/init_step": priming_step,
                })

                episode_log.reset()
                batch_processor.on_episode_start()
                step0_episode_idx += 1
                done = False
                action_type = "policy"
                action_plan.clear()
                # Always reset (leaves env ready either for this pass's next
                # fixed-seed episode, or for the main training loop right
                # after this block, which expects a valid post-reset
                # observation) -- seeded only while there's a next episode
                # of THIS pass still to run.
                if step0_episode_idx < success_rate_window:
                    env.reset(seed=int(step0_seeds[step0_episode_idx]))
                else:
                    env.reset()
                observation = env.get_observation()

        priming_pbar.close()
        wandb.log({"eval/init_progress": 1.0, "eval/init_step": priming_step})

        step0_success_rate = float(np.mean(training_log._success_window))
        wandb.log({"eval/success_rate": step0_success_rate, "training/global_step": 0})
        if eval_env is not None:
            # The first len(eval_seeds) episodes of this pass ran on the evaluation seeds (get_step0_seeds): the step-0
            # value is measured on the same episodes as every later evaluation. The policy is the frozen base VLA
            # (only_base_actions=True), whose flow-matching sampling is stochastic: step 0 is not a deterministic number.
            n_eval = len(eval_seeds)
            first = np.asarray(list(training_log._success_window)[:n_eval], dtype=float)
            rate0 = float(first.mean())
            stderr0 = float(first.std(ddof=1) / np.sqrt(n_eval)) if n_eval > 1 else 0.0
            wandb.log({
                "eval_rigorous/success_rate": rate0,
                "eval_rigorous/success_rate_stderr": stderr0,
                "eval_rigorous/n_episodes": n_eval,
                "training/global_step": 0,
            })
            logging.info(f"[step-0] eval_rigorous/success_rate={rate0:.3f} +/- {stderr0:.3f} on the {n_eval} evaluation "
                         f"seeds (first episodes of this same pass, frozen base policy, stochastic).")
            append_rigorous_eval_record(rl_eval_log_path, {
                "step": 0, "training_episode": 0, "policy": "frozen base VLA (only_base_actions, stochastic)",
                "critic_only": False, "n_episodes": n_eval, "success_rate": rate0, "stderr": stderr0,
                "seconds": None, "seeds": [int(x) for x in eval_seeds], "successes": [bool(x) for x in first],
                "episode_steps": None, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        logging.info(f"[step-0] Done after {priming_step} raw steps -- {success_rate_window} episodes "
                     f"evaluated, starting success_rate={step0_success_rate:.3f}. Real "
                     f"training begins now, budget (cfg.max_steps={cfg.max_steps}) untouched by this phase. "
                     f"eval/success_rate and training/* now use training/global_step as their own x-axis, "
                     f"starting fresh at 0 -- checkpoint step "
                     f"numbers (i) are NOT offset and remain exactly comparable across runs.")

    def run_periodic_rigorous_eval(step, agent_now):
        """Deterministic fixed-seed evaluation of `agent_now` on eval_env (see run_rigorous_eval), logged to wandb, to
        the log and to eval_rigorous.jsonl. Never raises: a failed evaluation is logged and training carries on."""
        eval_start = time.time()
        records = []
        try:
            success_rate, stderr = run_rigorous_eval(agent_now, eval_env, eval_seeds, cfg, episode_records=records)
            seconds = time.time() - eval_start
            wandb.log({
                "eval_rigorous/success_rate": success_rate,
                "eval_rigorous/success_rate_stderr": stderr,
                "eval_rigorous/n_episodes": len(eval_seeds),
                "eval_rigorous/seconds": seconds,
                "training/global_step": step,
            })
            logging.info(f"[rigorous-eval] step {step}: success_rate={success_rate:.3f} +/- {stderr:.3f} "
                         f"({len(eval_seeds)} episodes, {seconds:.0f}s)")
            append_rigorous_eval_record(rl_eval_log_path, {
                "step": step, "training_episode": training_log.ep_count,
                "policy": "trained agent, deterministic residual (critic argmax over base + edited candidates)",
                "critic_only": False, "n_episodes": len(eval_seeds), "success_rate": success_rate, "stderr": stderr,
                "seconds": seconds, "seeds": [r["seed"] for r in records], "successes": [r["success"] for r in records],
                "episode_steps": [r["steps"] for r in records], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception:
            logging.exception(f"[rigorous-eval] Failed at step {step}")

        if rl_eval_critic_only:
            # Critic-only variant: agent.replace() is a cheap struct copy (n_edit_samples is a static, non-pytree
            # field) -- the real agent is completely unaffected. Same seeds as the call above.
            critic_start = time.time()
            records = []
            try:
                critic_only_agent = agent_now.replace(n_edit_samples=0)
                critic_only_success_rate, critic_only_stderr = run_rigorous_eval(
                    critic_only_agent, eval_env, eval_seeds, cfg, episode_records=records
                )
                wandb.log({
                    "eval_rigorous/success_rate_critic_only": critic_only_success_rate,
                    "eval_rigorous/success_rate_critic_only_stderr": critic_only_stderr,
                    "training/global_step": step,
                })
                logging.info(
                    f"[rigorous-eval-critic-only] step {step}: "
                    f"success_rate={critic_only_success_rate:.3f} +/- {critic_only_stderr:.3f}"
                )
                append_rigorous_eval_record(rl_eval_log_path, {
                    "step": step, "training_episode": training_log.ep_count,
                    "policy": "trained agent, critic only (n_edit_samples=0)",
                    "critic_only": True, "n_episodes": len(eval_seeds), "success_rate": critic_only_success_rate,
                    "stderr": critic_only_stderr, "seconds": time.time() - critic_start,
                    "seeds": [r["seed"] for r in records], "successes": [r["success"] for r in records],
                    "episode_steps": [r["steps"] for r in records], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            except Exception:
                logging.exception(f"[rigorous-eval-critic-only] Failed at step {step}")

    # Resume-safe trigger bookkeeping (single-simulation servers): an evaluation is due once step // interval gets
    # beyond the bucket of the last one done / of the step the run (re)started from.
    last_rigorous_eval_bucket = (start_step // rl_eval_interval) if rl_eval_interval > 0 else 0

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
            action_chunk = as_env_chunk(action_chunk, observation, cfg)
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
            _last_si = episode_log.sample_info_history[-1] if episode_log.sample_info_history else None
            transition_dict = dict(
                observations=observation,
                actions=real_action,
                rewards=reward,
                masks=mask,
                dones=done,
                is_hil=(action_type == "human"),
            )
            if _last_si and "base_action" in _last_si:
                transition_dict["base_actions"] = _last_si["base_action"]
            batch_processor.insert_transition(transition_dict)
        
        can_update = training_log.ep_count >= 10 and i >= cfg.batch_size
        # On-policy (PPO/GRPO): update exactly when one full fresh rollout has
        # been collected, then discard it — NOT on the step/episode/batch
        # cadence below, which pairs with uniform random replay sampling and
        # would here either update on stale data or on a variable-length
        # (recompilation-triggering) batch. cfg.update_type/num_updates are
        # deliberately ignored in this branch.
        if is_on_policy_algo:
            if can_update and batch_processor.rollout_ready():
                run_agent_updates(1, step_metrics)
        elif cfg.update_type == "step" and can_update:
            run_agent_updates(cfg.num_updates, step_metrics)

        if done:
            batch_processor.on_episode_done(success)

            # Isaac Lab / FORGE: the evaluation shares the training simulation, so it runs HERE, between two
            # training episodes: the finished training episode is over, nothing of the next one has started (the
            # training env.reset() just below gives it a fresh start, as after any episode). It evaluates the agent
            # as of the last completed update. The time it takes is taken out of this step's loop time.
            if eval_env is not None and single_sim_env and rigorous_eval_due(
                    i, rl_eval_interval, last_rigorous_eval_bucket, True, True):
                last_rigorous_eval_bucket = i // rl_eval_interval
                _eval_t0 = time.time()
                run_periodic_rigorous_eval(i, agent)
                loop_start += time.time() - _eval_t0

            env.reset()

            if is_on_policy_algo:
                pass  # rollout-driven, handled above
            elif cfg.update_type == "episode" and can_update:
                for _ in tqdm.tqdm(range(cfg.num_updates)):
                    run_agent_updates(1, step_metrics)
            elif cfg.update_type == "batch" and can_update:
                episodes_since_update += 1
                if episodes_since_update >= cfg.num_batch:
                    for _ in tqdm.tqdm(range(cfg.num_updates)):
                        run_agent_updates(1, step_metrics)
                    episodes_since_update = 0

            training_log.on_episode_done(episode_log, success, step_metrics)
            step_metrics["training/episode_count"] = training_log.ep_count
            
            # Rolling success rate over the last success_rate_window episodes
            # (YAML-driven, computed once before the priming phase above --
            # shared with it so both stay in sync). Nothing is logged under
            # eval/success_rate at all until the window is genuinely full: a
            # partial window (e.g. after 1 episode) can ONLY read exactly
            # 0.0 or 1.0 regardless of the true underlying rate -- not a
            # calculation bug, just what a 1-sample mean is mathematically
            # forced to be. In practice the window is already full by the
            # time this loop starts (see the priming phase above), except
            # right when resuming a run whose in-memory window was never
            # reconstructed -- this guard still protects that case too.
            training_log._success_window.append(float(success))
            if len(training_log._success_window) > success_rate_window:
                training_log._success_window.pop(0)
            if len(training_log._success_window) >= success_rate_window:
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
                # save_checkpoint() only blocks for the synchronous portion
                # of the save (per Orbax's own docs: "Finished blocking save
                # in N seconds. Continuing to save asynchronously...") --
                # the background thread doing the actual host transfer can
                # still be running, and still using GPU memory for that
                # transfer, well after this call returns and this log line
                # prints. That transfer memory being unexpectedly still
                # resident is a plausible explanation for the OOM crashes
                # observed shortly after a checkpoint save reports complete.
                # wait_until_finished() is Orbax's own documented API for
                # genuinely blocking until that background thread is done
                # ("Blocks until any incomplete save operations are
                # completed... will wait until each of these checkpointers
                # is finished") -- calling it here, then gc.collect()
                # immediately after, gives that transfer memory a real
                # chance to be released before the next training step needs
                # its own peak memory.
                checkpoint_manager.wait_until_finished()
                gc.collect()
                logging.info(f"Saved agent checkpoint at step {i} (interval={cfg.checkpoint_interval})")
            except Exception as e:
                logging.error(f"Could not save model checkpoint: {e}")

        # Periodic rigorous deterministic eval (Jesse's proposed protocol) -- see run_periodic_rigorous_eval() above the
        # loop. Environments with their own simulation (ManiSkill): exactly here, at the multiples of rl_eval_interval.
        # Isaac Lab / FORGE (single simulation): never here, only between two episodes (the `if done:` block).
        if eval_env is not None and not single_sim_env and rigorous_eval_due(i, rl_eval_interval, 0, False, False):
            run_periodic_rigorous_eval(i, agent)

        if cfg.checkpoint_buffer and (has_action or action_type == "human"):
            try:
                save_replay_buffer_transition(checkpoint_dir_path, transition_dict, step=i)
            except Exception:
                logging.exception("Could not save agent buffer.")

        step_metrics["training/loop_time_ms"] = (time.time() - loop_start) * 1000.0
        step_metrics["training/global_step"] = i

        # TensorBoard logging — convert to float() to handle JAX scalars
        # (jnp.float32 etc.) which don't pass isinstance(v, (int, float))
        for k, v in step_metrics.items():
            try:
                tb_writer.add_scalar(k, float(v), global_step=i)
            except (TypeError, ValueError):
                pass  # skip non-scalar values (e.g. batch_info dicts)
        # No explicit step= here -- training/global_step (defined above via
        # wandb.define_metric) is what every training/eval chart actually
        # uses for its x-axis, starting fresh at 0 regardless of how long
        # priming took. See this file's priming-phase comment for why this
        # avoids the step-monotonicity data-loss issue entirely.
        wandb.log(step_metrics)
    
    if cfg.checkpoint_model:
        try:
            save_checkpoint(checkpoint_manager, agent, cfg.max_steps)
            logging.info(f"Saved final agent checkpoint at step {cfg.max_steps}")
        except Exception as e:
            logging.error(f"Could not save final checkpoint: {e}")
        logging.info("Waiting for checkpoint manager to finish")
        tb_writer.close()
        checkpoint_manager.wait_until_finished()
        # NOTE: the automatic end-of-training eval_curve.py sweep (writing a
        # handoff file for job_rl.sh to pick up) has been removed here --
        # eval/success_rate is now primed from step 1 (see the priming phase
        # above) and trustworthy throughout, so an automatic full 200-seed
        # sweep after every single run is no longer needed by default. The
        # scripts themselves (expo_ft/utils/eval_curve_runner.py,
        # scripts/run_eval_curve_from_handoff.py, scripts/eval_curve.py) are
        # untouched -- run scripts/eval_curve.py directly (see job_eval_curve.sh)
        # whenever a rigorous fixed-seed sweep is specifically wanted.


if __name__ == "__main__":
    app.run(main)
