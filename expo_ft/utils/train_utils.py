"""Shared utilities for Pi robot training scripts."""

import dataclasses
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from openpi.training import config as openpi_config


def init_logging() -> None:
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(checkpoint_dir: epath.Path, resuming: bool, project: str, name: str) -> None:
    wandb_id_file = checkpoint_dir / "wandb_id.txt"
    if resuming and wandb_id_file.exists():
        run_id = wandb_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=project, name=name)
    else:
        wandb.init(project=project, name=name)
        wandb_id_file.write_text(wandb.run.id)


def clear_batch(batch: Dict[str, Any]) -> None:
    """Recursively clear a batch dictionary to free memory."""
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, dict):
                clear_batch(v)
        batch.clear()


def get_batch_info(batch: Dict[str, Any]) -> Dict[str, float]:
    """Extract basic statistics from a batch dictionary for logging."""
    return {
        "rewards_mean": float(np.mean(batch["rewards"])),
        "rewards_std": float(np.std(batch["rewards"])),
        "rewards_max": float(np.max(batch["rewards"])),
        "rewards_min": float(np.min(batch["rewards"])),
        "masks_mean": float(np.mean(batch["masks"])),
        "masks_std": float(np.std(batch["masks"])),
        "valids_mean": float(np.mean(batch["valids"])),
        "valids_std": float(np.std(batch["valids"])),
        "actions_mean": float(np.mean(batch["actions"])),
        "actions_std": float(np.std(batch["actions"])),
        "actions_max": float(np.max(batch["actions"])),
        "actions_min": float(np.min(batch["actions"])),
        "states_mean": float(np.mean(batch["state"])),
        "states_std": float(np.std(batch["state"])),
        "states_max": float(np.max(batch["state"])),
        "states_min": float(np.min(batch["state"])),
        "base_image_max": float(np.max(batch["image"]["base_0_rgb"])),
        "base_image_min": float(np.min(batch["image"]["base_0_rgb"])),
        "base_image_std": float(np.std(batch["image"]["base_0_rgb"])),
    }


def list_saved_norm_stats(root):
    """Normalization statistics stored under <root>/assets/<asset_id>/norm_stats.json.

    openpi saves them next to the weights of every SFT checkpoint (<checkpoint>/assets/<asset_id>/), and
    setup_rl_norm_stats() puts a copy in every RL run folder. Returns (assets_dir, [asset_id, ...]); the list
    is empty when there is no such file.
    """
    assets = Path(root) / "assets"
    ids = sorted(d.name for d in assets.iterdir() if (d / "norm_stats.json").is_file()) if assets.is_dir() else []
    return assets, ids


def setup_rl_norm_stats(config, run_dir):
    """Make an RL run normalize with the statistics its SFT checkpoint was trained with.

    The RL actor, the replay buffers and the evaluation all build their normalization from
    config.pi05_assets_dir / config.pi05_asset_id (build_pi05_config). Left empty, they fall back to the openpi
    config's default (the official DROID statistics): wrong for an SFT trained on other statistics (for example
    the ones computed on the demos), since the model would then see differently scaled states and produce
    differently scaled actions than during its SFT.

    Source, by priority:
      1. config.pi05_assets_dir / pi05_asset_id already set (--config.pi05_assets_dir=... on the command line);
      2. <run_dir>/assets/<asset_id>/norm_stats.json (a resumed RL run: the copy made at its first start);
      3. <sft checkpoint>/assets/<asset_id>/norm_stats.json, the checkpoint given by
         config.pi05_weight_loader_path (<checkpoint>/params).
    Otherwise nothing is changed (config default, DROID statistics, as before).

    The statistics used are copied to <run_dir>/assets/<asset_id>/norm_stats.json, so that the evaluation of any
    checkpoint of this run (scripts/eval_policy.py --rl-checkpoint) and a resumed run find them without being told.
    A resumed run whose SFT checkpoint holds other statistics than the run's copy is refused.
    """
    run_dir = Path(run_dir)
    run_assets, run_ids = list_saved_norm_stats(run_dir)

    sft_assets, sft_ids = None, []
    if config.pi05_weight_loader_path:
        loader = Path(config.pi05_weight_loader_path)
        sft_assets, sft_ids = list_saved_norm_stats(loader.parent if loader.name == "params" else loader)

    if config.pi05_assets_dir or config.pi05_asset_id:
        origin = "set explicitly (--config.pi05_assets_dir / pi05_asset_id)"
    elif len(run_ids) == 1:
        config.pi05_assets_dir, config.pi05_asset_id = str(run_assets), run_ids[0]
        origin = "copy in the RL run folder (resumed run)"
        if len(sft_ids) == 1:
            a = (run_assets / run_ids[0] / "norm_stats.json").read_bytes()
            b = (sft_assets / sft_ids[0] / "norm_stats.json").read_bytes()
            if a != b:
                raise SystemExit(
                    f"FATAL: this RL run was started with the normalization statistics {run_assets / run_ids[0]}, "
                    f"but the SFT checkpoint now given holds different ones ({sft_assets / sft_ids[0]}). "
                    "Resume with the same SFT checkpoint (or without --sft-checkpoint)."
                )
    elif len(sft_ids) == 1:
        config.pi05_assets_dir, config.pi05_asset_id = str(sft_assets), sft_ids[0]
        origin = "saved with the SFT checkpoint"
    else:
        print(f"RL normalization statistics: config default (DROID); none saved with the SFT checkpoint "
              f"(found {sft_ids}) nor in the run folder (found {run_ids}).")
        return

    src = Path(config.pi05_assets_dir) / (config.pi05_asset_id or "") / "norm_stats.json"
    if not src.is_file():
        if origin.startswith("set explicitly"):
            # may legitimately be a remote path (gs://...) or one openpi resolves itself: do not copy, do not refuse
            print(f"RL normalization statistics: {src} ({origin}); not a local file, not copied to the run folder.")
            return
        raise SystemExit(f"FATAL: normalization statistics not found: {src} ({origin})")
    print(f"RL normalization statistics: {src} ({origin})")
    dst = run_assets / config.pi05_asset_id / "norm_stats.json"
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_pi05_config(config):
    """Extract pi05 settings from agent config and build the openpi train config.

    Returns (agent_kwargs, pi05_train_config, pi05_resize_size, model_cls).
    ``agent_kwargs`` is a plain dict with pi05-specific keys removed.
    """
    agent_kwargs = dict(config)
    pi05_config_name = agent_kwargs.pop("pi05_config_name")
    pi05_resize_size = agent_kwargs.pop("pi05_resize_size")
    pi05_weight_loader_path = agent_kwargs.pop("pi05_weight_loader_path", "") or None
    pi05_assets_dir = agent_kwargs.pop("pi05_assets_dir", "") or None
    pi05_asset_id = agent_kwargs.pop("pi05_asset_id", "") or None
    model_cls = agent_kwargs.pop("model_cls")

    pi05_train_config = openpi_config.get_config(
        pi05_config_name, weight_loader_path=pi05_weight_loader_path
    )
    if pi05_assets_dir or pi05_asset_id:
        from openpi.training.config import AssetsConfig
        new_assets = AssetsConfig(
            assets_dir=pi05_assets_dir or pi05_train_config.data.assets.assets_dir,
            asset_id=pi05_asset_id or pi05_train_config.data.assets.asset_id,
        )
        pi05_train_config = dataclasses.replace(
            pi05_train_config,
            data=dataclasses.replace(pi05_train_config.data, assets=new_assets),
        )
    return agent_kwargs, pi05_train_config, pi05_resize_size, model_cls


def _concat_leaves(x, y):
    """tree_map'd concatenation along axis 0 that tolerates None leaves."""
    if x is None:
        return y
    if y is None:
        return x
    return jnp.concatenate([jnp.asarray(x), jnp.asarray(y)], axis=0)


def _shuffle_batch(key, batch):
    """Shuffle a (possibly nested) batch along axis=0 with one shared permutation."""
    leaves, treedef = jax.tree_util.tree_flatten(batch)
    n = leaves[0].shape[0]
    perm = jax.random.permutation(key, n)
    shuffled = [jnp.asarray(x)[perm] for x in leaves]
    return jax.tree_util.tree_unflatten(treedef, shuffled)


def combine_batches(online_batch, offline_batch, rng):
    """Combine online and offline batches: concatenate, then shuffle to mix.

    Batches must already have the desired sizes (proportional to offline_ratio).
    """
    combined = jax.tree_util.tree_map(_concat_leaves, online_batch, offline_batch)
    return _shuffle_batch(rng, combined)
