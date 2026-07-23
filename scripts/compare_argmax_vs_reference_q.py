"""
Diagnostic suggested by Albert: for a batch of real demo states, compare
the Q value our current agent's critic assigns to (a) whatever action its
own argmax-over-candidates picks, vs (b) the action a known-good reference
policy checkpoint (e.g. a 96%-SR SFT checkpoint) would take for the same
state. If the critic consistently rates its own pick HIGHER than the
reference action, that's direct evidence its RELATIVE ranking of
candidates is broken -- not just an absolute-scale overestimation issue
(which is what target_q_max climbing shows on its own).

Two independent π₀.₅ instances are built: the one frozen inside the RL
checkpoint (whatever it was actually trained against), and a separate one
loaded fresh from the reference checkpoint's own weights -- these are two
different sets of weights, not the same model reused.

Accepts MULTIPLE --rl-checkpoint values (e.g. an early and a late step from
the same run) in one process: demo data, the reference checkpoint, and the
sampled batch of states are all loaded/drawn ONCE and reused across every
RL checkpoint -- only the RL checkpoint's own weights get reloaded per
value, which is both cheaper and a fairer apples-to-apples comparison
(same states scored against each checkpoint, not a fresh random draw per
run).

Usage:
    python scripts/compare_argmax_vs_reference_q.py \
        --config configs/task/maniskill/push_cube_expo_ft.yaml \
        --rl-checkpoint logs/push_cube/<run>/checkpoints/20000 \
        --rl-checkpoint logs/push_cube/<run>/checkpoints/118000 \
        --reference-checkpoint /path/to/96pct/params \
        --n-states 100 \
        --output-json q_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to task YAML (e.g. configs/task/maniskill/push_cube_expo_ft.yaml)")
    parser.add_argument("--rl-checkpoint", required=True, action="append", help="Path to an RL/EXPOLearner checkpoint STEP directory (e.g. .../checkpoints/40000) -- the critic being examined. Repeat this flag to evaluate multiple checkpoints (e.g. an early and a late step) in one process, reusing everything else.")
    parser.add_argument("--reference-checkpoint", required=True, help="Path to the reference SFT checkpoint's 'params' directory (e.g. the 96%% SR checkpoint) -- NOT the RL checkpoint's own frozen VLA")
    parser.add_argument("--n-states", type=int, default=200, help="Number of demo states to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None, help="Optional path to dump per-state results + summary (per checkpoint) as JSON")
    parser.add_argument("--output-csv", default=None, help="Optional path to dump a long-format CSV (one row per state per checkpoint: rl_checkpoint, rl_step, state_idx, q_agent_pick, q_reference, gap) -- the easiest format to load into pandas/matplotlib/a wandb Table for plotting")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import openpi.training.sharding as openpi_sharding
    from expo_ft.utils.config_loader import load_task_config, get_sft_config_name
    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.env.droid_utils import process_droid_dataset
    from expo_ft.agents.alg.batch_utils import prepare_critic_batch, prepare_actor_sampling_batch_current

    cfg = load_task_config(args.config)
    cfg.normalize_action = True

    if getattr(cfg, "model_cls", "EXPOLearner") == "EXPOLearnerOld":
        from expo_ft.agents.alg.expo_ft_old import load_agent, restore_checkpoint
    else:
        from expo_ft.agents.alg.expo_ft import load_agent, restore_checkpoint

    mesh = openpi_sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ── Real demo data (not the synthetic shapes-only dataset eval_policy.py
    # uses -- we need genuine states here, not just an env to roll out in).
    # Loaded ONCE, reused for every RL checkpoint below. ──
    print(f"Loading demo data from {cfg.droid_format_dir} ...")
    dataset = process_droid_dataset(cfg.droid_format_dir, cfg, num_data=None)
    example_action = dataset[0]['actions'][np.newaxis]

    replay_buffer = create_replay_buffer(
        config=cfg,
        example_action=example_action,
        capacity=len(dataset),
        task_description=cfg.language_instruction,
        replan_steps=cfg.replan_steps,
        seed=args.seed,
    )
    replay_buffer.insert_dataset(dataset)

    agent_example_observation, agent_example_state, agent_example_action = \
        replay_buffer.convert_to_critic_format({
            "base_image":       replay_buffer.dataset_dict['base_image'][0][np.newaxis],
            "left_wrist_image": replay_buffer.dataset_dict['left_wrist_image'][0][np.newaxis],
            "state":            replay_buffer.dataset_dict['state'][0][np.newaxis],
            "actions":          replay_buffer.dataset_dict['actions'][0][np.newaxis],
        })

    # ── Load model config for the RL checkpoints' architecture (used to
    # build both the RL agent structure and the reference checkpoint below --
    # same pi05 config, just different weights). ──
    import importlib.util
    model_cls_name = getattr(cfg, "model_cls", "EXPOLearner")
    model_config_path = REPO_ROOT / (
        "configs/model/expo_ft_old_pi_config.py" if model_cls_name == "EXPOLearnerOld" else "configs/model/expo_ft_pi_config.py"
    )
    spec = importlib.util.spec_from_file_location("model_config", str(model_config_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # ── Build the RL agent STRUCTURE once (architecture only -- restore_checkpoint
    # below overwrites the actual weights per checkpoint, so this build is just a
    # structural placeholder, same as eval_policy.py). Reused across all --rl-checkpoint values. ──
    model_config = mod.get_config()
    model_config.pi05_config_name = get_sft_config_name(cfg)
    model_config.skip_repack_transforms = cfg.skip_repack_transforms
    model_config.pi05_weight_loader_path = None

    print("Building RL agent structure...")
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        model_config, args.seed, mesh, data_sharding, replicated_sharding,
        resume=False, default_prompt=cfg.language_instruction,
    )
    actor.action_dim = agent_example_action.squeeze().shape[-1]
    actor.state_dim = agent_example_state.squeeze().shape[-1]

    base_agent = load_agent(
        seed=args.seed,
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
        resume=False,
        replan_steps=cfg.replan_steps,
        default_prompt=cfg.language_instruction,
        residual_action_xyzg=getattr(cfg, 'residual_action_xyzg', False),
    )

    # ── Reference checkpoint: a SEPARATE π₀.₅ instance, own weights, own
    # train_state -- independent of any RL agent's frozen VLA. Built ONCE,
    # reused across all --rl-checkpoint values. ──
    print(f"Loading reference checkpoint from {args.reference_checkpoint} ...")
    ref_model_config = mod.get_config()
    ref_model_config.pi05_config_name = get_sft_config_name(cfg)
    ref_model_config.skip_repack_transforms = cfg.skip_repack_transforms
    ref_model_config.pi05_weight_loader_path = str(Path(args.reference_checkpoint))
    reference_actor, reference_actor_train_state, _, _, _ = build_pi05(
        ref_model_config, args.seed, mesh, data_sharding, replicated_sharding,
        resume=False, default_prompt=cfg.language_instruction,
    )
    reference_actor.action_dim = agent_example_action.squeeze().shape[-1]
    reference_actor.state_dim = agent_example_state.squeeze().shape[-1]

    # ── Pull a batch of real demo states ONCE -- the SAME states get scored
    # against every RL checkpoint below, for a fair apples-to-apples comparison. ──
    print(f"Sampling {args.n_states} demo states...")
    iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": args.n_states}, data_sharding=data_sharding,
    )
    raw_batch = next(iterator)
    raw_batch = dict(raw_batch)
    batch = prepare_critic_batch(
        raw_batch,
        base_agent.actor.model_config.action_dim,
        base_agent.action_dim,
        base_agent.state_dim,
        base_agent.action_horizon,
        base_agent.replan_steps,
    )
    batch_size = batch["states"].shape[0]

    diag_batch = dict(batch)
    diag_batch["next_image"] = raw_batch["image"]
    diag_batch["next_image_mask"] = raw_batch["image_mask"]
    diag_batch["next_states"] = batch["states"]
    diag_batch["next_critic_states"] = batch["critic_states"]
    diag_batch["next_observations"] = batch["observations"]

    # ── Reference action doesn't depend on which RL checkpoint is being
    # examined -- compute it ONCE, reuse for every checkpoint below. ──
    key = jax.random.PRNGKey(args.seed + 1)
    transformed_inputs = prepare_actor_sampling_batch_current(raw_batch)
    transformed_inputs = dict(transformed_inputs)
    transformed_inputs["state"] = batch["states"]
    reference_actions, _ = reference_actor.sample_training_actions(
        transformed_inputs=transformed_inputs,
        train_state=reference_actor_train_state,
        rng=key,
        train=False,
        num_samples=1,
    )
    reference_actions = reference_actions[:, :base_agent.replan_steps, :]
    reference_actions = reference_actions.reshape(batch_size, -1)

    import orbax.checkpoint as ocp
    all_results = {}
    csv_rows = []  # long format: one row per (checkpoint, state) -- easiest thing to plot
    checkpoint_managers = {}  # cache by checkpoint_dir so repeated steps from the same run don't rebuild it

    for rl_checkpoint_str in args.rl_checkpoint:
        rl_ckpt_path = Path(rl_checkpoint_str).resolve()
        rl_step = int(rl_ckpt_path.name)
        rl_checkpoint_dir = rl_ckpt_path.parent

        if rl_checkpoint_dir not in checkpoint_managers:
            checkpoint_managers[rl_checkpoint_dir] = ocp.CheckpointManager(
                rl_checkpoint_dir,
                item_handlers={"agent": ocp.PyTreeCheckpointHandler(), "params": ocp.PyTreeCheckpointHandler()},
                options=ocp.CheckpointManagerOptions(create=False),
            )
        rl_mngr = checkpoint_managers[rl_checkpoint_dir]

        print(f"Restoring RL checkpoint: step {rl_step} from {rl_checkpoint_dir}")
        agent = restore_checkpoint(rl_mngr, base_agent, rl_step)
        agent = agent.cache_infer_params()

        # ── (a) What does OUR pipeline actually pick right now for these
        # states? Reuses sample_batch_actions verbatim (same argmax/candidate
        # logic used during real training) rather than reimplementing it. ──
        agent_actions, sample_info, _ = agent.sample_batch_actions(diag_batch)
        agent_actions = agent_actions.reshape(batch_size, -1)

        # ── Query THIS checkpoint's critic on both action sets, at the
        # current state -- no next_-hack needed here, batch['observations']
        # /['critic_states'] are already the genuine current-state fields. ──
        q_agent_pick = agent._compute_q_split(
            agent.target_critic.apply_fn, agent.target_critic.params, agent.target_critic.batch_stats,
            batch["observations"], agent_actions, batch["critic_states"],
        )
        q_reference = agent._compute_q_split(
            agent.target_critic.apply_fn, agent.target_critic.params, agent.target_critic.batch_stats,
            batch["observations"], reference_actions, batch["critic_states"],
        )

        q_agent_pick = np.asarray(q_agent_pick).reshape(-1)
        q_reference = np.asarray(q_reference).reshape(-1)
        gap = q_agent_pick - q_reference
        misrank_rate = float(np.mean(gap > 0))

        # Short, sortable key for both the JSON dict and CSV rows -- the
        # step number alone (e.g. "20000"), not the full path, so results
        # are easy to skim/sort/plot by. Full path kept in the summary
        # itself in case two different runs happen to share a step number.
        ckpt_key = str(rl_step)

        summary = {
            "rl_checkpoint_path": rl_checkpoint_str,
            "rl_step": rl_step,
            "n_states": int(batch_size),
            "q_agent_pick_mean": float(np.mean(q_agent_pick)),
            "q_agent_pick_median": float(np.median(q_agent_pick)),
            "q_reference_mean": float(np.mean(q_reference)),
            "q_reference_median": float(np.median(q_reference)),
            "gap_mean": float(np.mean(gap)),
            "gap_median": float(np.median(gap)),
            "gap_std": float(np.std(gap)),
            "misrank_rate": misrank_rate,  # fraction of states where the critic rates its OWN pick higher than the reference action
        }
        print(json.dumps(summary, indent=2))
        all_results[ckpt_key] = {
            "summary": summary,
            "per_state": {
                "q_agent_pick": q_agent_pick.tolist(),
                "q_reference": q_reference.tolist(),
                "gap": gap.tolist(),
            },
        }
        for state_idx in range(batch_size):
            csv_rows.append({
                "rl_checkpoint": rl_checkpoint_str,
                "rl_step": rl_step,
                "state_idx": state_idx,
                "q_agent_pick": float(q_agent_pick[state_idx]),
                "q_reference": float(q_reference[state_idx]),
                "gap": float(gap[state_idx]),
            })

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON to {args.output_json}")

    if args.output_csv:
        import csv as csv_module
        with open(args.output_csv, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=["rl_checkpoint", "rl_step", "state_idx", "q_agent_pick", "q_reference", "gap"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved CSV to {args.output_csv} ({len(csv_rows)} rows -- one per state per checkpoint, ready for pandas/matplotlib/a wandb Table)")


if __name__ == "__main__":
    main()
