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
