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

    # Convert nested lists to tuples where needed (e.g. rl_hidden_dims)
    if hasattr(cfg, "rl_hidden_dims"):
        cfg.rl_hidden_dims = tuple(cfg.rl_hidden_dims)

    return cfg


def resolve_run_dir(cfg: SimpleNamespace) -> tuple[str, bool]:
    """
    Resolve the run directory for this experiment.

    - If cfg.resume_dir is set: resume from that exact directory.
    - Otherwise: create a new timestamped subdirectory under cfg.output_dir.

    Returns:
        (run_dir, resuming) where resuming is True if we are resuming.
    """
    if cfg.resume_dir is not None:
        run_dir = cfg.resume_dir
        resuming = True
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.run_name}_{timestamp}"
        run_dir = os.path.join(cfg.output_dir, run_name)
        resuming = False

    os.makedirs(run_dir, exist_ok=True)
    return run_dir, resuming


def get_sft_config_name(cfg: SimpleNamespace) -> str:
    """
    Return the openpi TrainConfig name to use for SFT warmup,
    based on task config flags.
    """
    if cfg.sft_use_lora:
        return "expo_pi05_droid_lora_finetune_sft_cartesian_state"
    else:
        return "expo_pi05_droid_full_finetune_sft_cartesian_state"
