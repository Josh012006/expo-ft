"""Measure what the RL stage really needs (GPU memory, host memory, time per call) in a few minutes, BEFORE launching it.

No environment, no training loop, no wandb: it builds exactly what train_pi_robo.py builds (pi0.5 + residual actor +
critics + replay buffers + BatchProcessor, from the same task YAML through apply_task_overrides) and runs the real
programs on demonstration data:

    stage                what it runs                                   what it tells you
    0 start              nothing                                        CUDA context, what the GPU already holds
    1 vla                build_pi05                                     weights of pi0.5 (+ LoRA state)
    2 agent              load_agent                                     + encoder, critics, residual actor, optimizers
    3 sample_actions     N candidates + critic argmax, 1 obs (compile)  inference peak, latency of one chunk (= rollout)
    4 update (compile)   agent.update, both actor branches              compile-time peak of the update program
    5 update (steady)    --n-updates more agent.update                  steady peak, seconds per update

"peak" is JAX's own allocator peak (jax Device.memory_stats()['peak_bytes_in_use']): it only grows, so read it as the
running maximum up to that stage. "nvidia-smi used" is the whole process as the driver sees it (adds the CUDA context and
the cuBLAS / cuDNN workspaces that JAX does not count). The number to size a GPU with is the LAST "nvidia-smi used".

Preallocation is switched OFF (XLA_PYTHON_CLIENT_PREALLOCATE=false unless you set it yourself): with JAX's default the
process grabs 75 % (97 % in jobs/job_rl.sh) of the card at start and nvidia-smi shows that, not the need.

To answer "does it fit in N GB?" on a bigger card, cap it: XLA_PYTHON_CLIENT_MEM_FRACTION=<N / card size>. An
out-of-memory error is then the answer (stages already printed stay valid).

    sbatch jobs/job_profile_memory.sh --config configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml

Data volume is irrelevant for the GPU (a batch has a fixed size), so a handful of demonstration episodes is enough
(--num-demos); the replay buffers are small here (the real ones are max_steps rows each, in host RAM: see the host
estimate printed at the end).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # must be set before jax is imported

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

GB = 1024 ** 3


# ------------------------------------------------------------------ measurements
def host_rss_gb():
    """Resident set size of this process (GB), from /proc (no dependency)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return float("nan")


def nvidia_smi_used_gb(pid):
    """GPU memory the driver attributes to this process (GB); nan if nvidia-smi is not usable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        for line in out.strip().splitlines():
            p, used = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                return float(used) / 1024
    except Exception:
        pass
    # PID not listed (PID namespace): a Slurm job sees only its own GPU, so if exactly one GPU is visible its total
    # used memory is this process (the job has the card to itself).
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
        if len(out) == 1:
            return float(out[0]) / 1024
    except Exception:
        pass
    return float("nan")


class Recorder:
    def __init__(self, device, out_path):
        self.device, self.out_path, self.rows = device, out_path, []

    def stage(self, name, seconds=None, note=""):
        import jax
        jax.effects_barrier()
        stats = self.device.memory_stats() or {}
        row = {
            "stage": name,
            "jax_in_use_gb": stats.get("bytes_in_use", float("nan")) / GB,
            "jax_peak_gb": stats.get("peak_bytes_in_use", float("nan")) / GB,
            "jax_limit_gb": stats.get("bytes_limit", float("nan")) / GB,
            "nvidia_smi_used_gb": nvidia_smi_used_gb(os.getpid()),
            "host_rss_gb": host_rss_gb(),
            "seconds": seconds,
            "note": note,
        }
        self.rows.append(row)
        sec = "" if seconds is None else f"  {seconds:7.2f}s"
        print(f"[{name:<22}] jax in use {row['jax_in_use_gb']:6.2f} | jax peak {row['jax_peak_gb']:6.2f} | "
              f"nvidia-smi used {row['nvidia_smi_used_gb']:6.2f} | host RSS {row['host_rss_gb']:6.2f} GB{sec} {note}",
              flush=True)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.rows, indent=2))      # rewritten at each stage: survives a crash
        return row


def timed(fn):
    import jax
    t = time.time()
    out = fn()
    jax.block_until_ready(out)
    return out, time.time() - t


# ------------------------------------------------------------------ setup (same sequence as train_pi_robo.py)
def load_model_config(cfg):
    model_cls = getattr(cfg, "model_cls", "EXPOLearner")
    if model_cls not in ("EXPOLearner", "EXPOLearnerCategorical"):
        raise ValueError(f"profile_memory.py supports EXPOLearner / EXPOLearnerCategorical, got {model_cls}")
    path = REPO_ROOT / ("configs/model/expo_ft_categorical_pi_config.py" if model_cls == "EXPOLearnerCategorical"
                        else "configs/model/expo_ft_pi_config.py")
    spec = importlib.util.spec_from_file_location("model_config", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def main(args):
    import jax
    import openpi.training.sharding as openpi_sharding

    from expo_ft.agents.vla.pi05 import build_pi05
    from expo_ft.data.batch_processor import BatchProcessor
    from expo_ft.data.replay_buffer import create_replay_buffer
    from expo_ft.env.droid_utils import load_demo_dataset
    from expo_ft.utils.config_loader import apply_task_overrides, load_task_config
    from expo_ft.utils.train_utils import list_saved_norm_stats

    cfg = load_task_config(args.config)
    cfg.normalize_action = True
    model_config = load_model_config(cfg)
    aux_spec = apply_task_overrides(model_config, cfg)

    if getattr(cfg, "model_cls", "EXPOLearner") == "EXPOLearnerCategorical":
        from expo_ft.agents.alg.expo_ft_categorical import load_agent
    else:
        from expo_ft.agents.alg.expo_ft import load_agent

    device = jax.local_devices()[0]
    out_path = Path(args.output) if args.output else (
        REPO_ROOT / "logs" / "profile_memory" / f"{Path(args.config).stem}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json")
    rec = Recorder(device, out_path)

    print("=" * 100)
    print(f"config             {args.config}")
    print(f"device             {device} ({device.device_kind}), {jax.device_count()} device(s)")
    print(f"XLA preallocate    {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}   "
          f"mem fraction {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '(default)')}")
    print(f"model_cls          {model_config.model_cls}   N={model_config.N}  n_edit_samples={model_config.n_edit_samples}  "
          f"num_qs={model_config.num_qs}")
    print(f"batch_size={cfg.batch_size}  utd_ratio={cfg.utd_ratio}  offline_ratio={cfg.offline_ratio}  "
          f"replan_steps={cfg.replan_steps}  actor_success_only={getattr(cfg, 'actor_success_only', False)}")
    print(f"tactile / force    {aux_spec.describe()}")
    print("=" * 100, flush=True)
    rec.stage("0 start")

    mesh = openpi_sharding.make_mesh(getattr(cfg, "fsdp_devices", 1))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Weights: the architecture decides the memory, not the values, so the base pi0.5 is enough; an SFT checkpoint
    # gives the same numbers (and is what the real run uses).
    if args.checkpoint:
        model_config.pi05_weight_loader_path = str(Path(args.checkpoint) / "params")
        assets, ids = list_saved_norm_stats(args.checkpoint)
        if len(ids) == 1:
            model_config.pi05_assets_dir = str(assets)
            model_config.pi05_asset_id = ids[0]
    else:
        model_config.pi05_weight_loader_path = None

    dataset = load_demo_dataset(cfg, num_data=args.num_demos)
    example_action = dataset[0]["actions"][np.newaxis]
    print(f"{len(dataset)} demonstration steps loaded", flush=True)

    # 1 -- the frozen VLA (+ LoRA state)
    t = time.time()
    actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
        model_config, cfg.seed, mesh, data_sharding, replicated_sharding, resume=False,
        default_prompt=cfg.language_instruction)
    jax.block_until_ready(actor_train_state)
    rec.stage("1 vla", time.time() - t)

    # replay buffers: small (capacity is host RAM only, the GPU only ever sees fixed-size batches)
    capacity = len(dataset) + cfg.replan_steps + 8
    rb_args = dict(config=model_config, example_action=example_action, capacity=capacity,
                   task_description=cfg.language_instruction, replan_steps=cfg.replan_steps, seed=cfg.seed)
    replay_buffer = create_replay_buffer(**rb_args)
    offline_replay_buffer = create_replay_buffer(**rb_args)
    batch_processor = BatchProcessor(
        replay_buffer=replay_buffer,
        offline_replay_buffer=offline_replay_buffer,
        data_sharding=data_sharding,
        batch_size=cfg.batch_size,
        utd_ratio=cfg.utd_ratio,
        offline_ratio=cfg.offline_ratio,
        actor_success_only=getattr(cfg, "actor_success_only", False),
        use_dagger_hil_sampling=False,
        dataset=dataset,
        seed_demos_online=True,        # the online buffer must hold something to sample from (shapes are what matter)
        replan_steps=cfg.replan_steps,
    )

    ex_obs, ex_state, ex_action = offline_replay_buffer.convert_to_critic_format({
        "base_image": offline_replay_buffer.dataset_dict["base_image"][0][np.newaxis],
        "left_wrist_image": offline_replay_buffer.dataset_dict["left_wrist_image"][0][np.newaxis],
        "state": offline_replay_buffer.dataset_dict["state"][0][np.newaxis],
        "actions": offline_replay_buffer.dataset_dict["actions"][0][np.newaxis],
    })
    actor.action_dim = ex_action.squeeze().shape[-1]
    actor.state_dim = ex_state.squeeze().shape[-1]

    # 2 -- critics, encoder, residual actor, optimizers
    t = time.time()
    agent = load_agent(
        seed=cfg.seed, example_observation=ex_obs.squeeze(), example_action=ex_action.squeeze(),
        example_state=ex_state.squeeze(), actor=actor, actor_train_state=actor_train_state,
        target_actor_params=target_actor_params, agent_kwargs=agent_kwargs, metadata=vla_metadata, mesh=mesh,
        data_sharding=data_sharding, replicated_sharding=replicated_sharding, resume=False,
        replan_steps=cfg.replan_steps, default_prompt=cfg.language_instruction,
        residual_action_xyzg=getattr(cfg, "residual_action_xyzg", False))
    jax.block_until_ready(agent)
    rec.stage("2 agent", time.time() - t)

    # 3 -- one rollout step: N candidates from the base policy + residual + critic argmax
    observation = dataset[0]["observations"]
    (out, seconds) = timed(lambda: agent.sample_actions(observation))
    action_chunk, agent, _ = out
    rec.stage("3a sample (compile)", seconds, f"chunk {np.asarray(action_chunk).shape}")
    times = []
    for _ in range(args.n_samples):
        (out, seconds) = timed(lambda: agent.sample_actions(observation))
        action_chunk, agent, _ = out
        times.append(seconds)
    rec.stage("3b sample (steady)", float(np.mean(times)), f"mean of {len(times)}  <- seconds per chunk, executed every "
                                                            f"{cfg.replan_steps} env steps")

    if args.skip_update:
        _finish(rec, cfg, model_config, args)
        return

    # 4 -- the update program, compiled with the real batch shapes, both actor branches
    combine_rng = jax.random.PRNGKey(cfg.seed + 100)
    batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
    print(f"update batch: critic batch {cfg.batch_size * cfg.utd_ratio} transitions "
          f"({cfg.utd_ratio} minibatches of {cfg.batch_size}); actor batch "
          f"{'none (no success transition in the buffers)' if actor_batch is None else cfg.batch_size}", flush=True)

    def one_update(agent, batch, actor_batch):
        agent = agent.replace(rng=jax.device_put(agent.rng, replicated_sharding))
        return agent.update(agent, batch, cfg.utd_ratio, actor_batch)

    (out, seconds) = timed(lambda: one_update(agent, batch, None))
    agent, _ = out
    rec.stage("4a update (compile)", seconds, "actor_batch=None branch")
    if actor_batch is not None:
        batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
        (out, seconds) = timed(lambda: one_update(agent, batch, actor_batch))
        agent, _ = out
        rec.stage("4b update (compile)", seconds, "success-only actor branch (the one training uses)")

    # 5 -- steady state
    times = []
    for _ in range(args.n_updates):
        batch, actor_batch, combine_rng = batch_processor.next_batch(combine_rng)
        (out, seconds) = timed(lambda: one_update(agent, batch, actor_batch))
        agent, _ = out
        times.append(seconds)
    rec.stage("5 update (steady)", float(np.mean(times)), f"mean of {len(times)}  <- seconds per agent.update "
                                                           f"(= {cfg.utd_ratio} critic steps + 1 actor step)")
    # the real loop also re-caches the inference params after each update (inside update); one more sample shows
    # the rollout cost once training has started
    (out, seconds) = timed(lambda: agent.sample_actions(observation))
    rec.stage("6 sample after update", seconds)
    _finish(rec, cfg, model_config, args)


def _finish(rec, cfg, model_config, args):
    last = rec.rows[-1]
    peak = max(r["jax_peak_gb"] for r in rec.rows)
    smi_values = [r["nvidia_smi_used_gb"] for r in rec.rows if r["nvidia_smi_used_gb"] == r["nvidia_smi_used_gb"]]
    smi = max(smi_values) if smi_values else float("nan")
    # host RAM of the real run: two replay buffers of max_steps rows (np.empty: only the rows written are touched,
    # but a full buffer is what a full run reaches). One row = 3 images uint8 + chunk + state + small fields.
    row_bytes = 3 * (224 * 224 * 3) + 16 * 32 * 4 + 32 * 4 + 4096
    max_steps = getattr(cfg, "max_steps", None)
    print("=" * 100)
    print(f"GPU  : JAX peak {peak:.1f} GB | nvidia-smi used (max over stages) {smi:.1f} GB   <- size the GPU with this "
          f"+ ~10 % margin")
    if max_steps:
        buf = 2 * max_steps * row_bytes / GB
        print(f"HOST : 2 replay buffers at max_steps={max_steps} rows ~ {buf:.0f} GB (full), + model restore / python "
              f"~ {last['host_rss_gb']:.0f} GB measured here  -> --mem >= {buf + last['host_rss_gb'] + 8:.0f} GB")
    print(f"saved: {rec.out_path}")
    print("Isaac Lab is NOT included: measure it separately (nvidia-smi per process, see jobs/job_profile_memory.sh).")
    print("=" * 100, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="task YAML of the RL run")
    parser.add_argument("--checkpoint", default=None, help="SFT checkpoint STEP directory (default: base pi0.5 weights)")
    parser.add_argument("--num-demos", type=int, default=5, help="demonstration episodes to load (default 5)")
    parser.add_argument("--n-samples", type=int, default=5, help="timed sample_actions calls (default 5)")
    parser.add_argument("--n-updates", type=int, default=3, help="timed steady-state agent.update calls (default 3)")
    parser.add_argument("--skip-update", action="store_true", help="stop after sample_actions (inference-only numbers)")
    parser.add_argument("--output", default=None, help="JSON report path (default logs/profile_memory/<config>_<time>.json)")
    main(parser.parse_args())
