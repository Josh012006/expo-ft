"""
Master pipeline script for EXPO-FT on ManiSkill tasks.

Reads a single task YAML config and orchestrates:
    1. compute_norm_stats  (openpi)
    2. SFT warmup          (openpi train.py)
    3. RL training         (train_pi_robo.py)

Usage:
    python scripts/run_pipeline.py --config configs/task/stack_cube.yaml --stage all
    python scripts/run_pipeline.py --config configs/task/stack_cube.yaml --stage norm_stats
    python scripts/run_pipeline.py --config configs/task/stack_cube.yaml --stage sft
    python scripts/run_pipeline.py --config configs/task/stack_cube.yaml --stage rl
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from expo_ft.utils.config_loader import load_task_config, resolve_run_dir, get_sft_config_name

# LeRobot dataset home — same as in convert_maniskill_to_lerobot.py
os.environ["HF_LEROBOT_HOME"] = str(REPO_ROOT / "demos" / "lerobot")

OPENPI_SCRIPTS = REPO_ROOT / "expo_ft" / "agents" / "vla" / "openpi" / "scripts"
TRAIN_PI_ROBO  = REPO_ROOT / "train_pi_robo.py"


def run(cmd: list[str], **kwargs):
    """Run a subprocess command, raise on failure."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def stage_demos(cfg, args):
    """Generate ManiSkill demos, replay-validate them in the task's control mode,
    then convert to droid_format (RL offline buffer) and LeRobot (SFT trainer)."""
    n_demos = getattr(cfg, "num_demos", 550)
    mp_root = REPO_ROOT / "demos"  # mani_skill appends <env_id>/motionplanning itself

    # 1. Generate raw mplib demos (native pd_joint_pos), keep only successful ones.
    #    Forced to physx_cpu (not cfg.sim_backend) to match step 2, which is hard-locked
    #    to CPU regardless (control mode conversion isn't supported on GPU) — this keeps
    #    the whole demo-prep pipeline on one consistent backend, no mid-pipeline switch.
    run([
        "python", "-m", "mani_skill.examples.motionplanning.panda.run",
        "-e", cfg.env_id, "-n", str(n_demos), "--only-count-success",
        "--record-dir", str(mp_root), "--traj-name", "trajectory", "-b", "physx_cpu",
    ])
    raw_h5 = mp_root / cfg.env_id / "motionplanning" / "trajectory.h5"

    # 2. Replay into RGB + the task's control mode. This IS the validity check —
    #    the printed "X/N=XX% demos saved" at the end is the demo success rate.
    #    NOTE: two GOTCHAS here, both hard requirements of ManiSkill's replay tool:
    #    (a) control mode conversion (native pd_joint_pos -> cfg.control_mode) is not
    #        supported on GPU-parallelized backends -> must force physx_cpu.
    #    (b) --use-env-states is INCOMPATIBLE with control mode conversion (ManiSkill
    #        asserts on this) since converting modes changes how many actions are
    #        needed for the same states, so state teleportation is meaningless here.
    #        Conversion instead genuinely re-simulates forward on CPU via
    #        action_conversion.from_pd_joint_pos — the resulting demos are real
    #        physx_cpu dynamics, not GPU-teleported states.
    run([
        "python", str(REPO_ROOT / "scripts" / "replay_trajectory_patched.py"),
        "--expo-config", args.config,
        "--traj-path", str(raw_h5), "--save-traj",
        "-o", "rgb", "-c", cfg.control_mode, "-b", "physx_cpu",
    ])
    rgb_h5 = raw_h5.parent / f"trajectory.rgb.{cfg.control_mode}.physx_cpu.h5"

    # 3. droid_format (used by stage_rl's offline buffer)
    run([
        "python", str(REPO_ROOT / "scripts" / "convert_maniskill_to_droid.py"),
        "--traj-path", str(rgb_h5), "--output-dir", cfg.droid_format_dir,
        "--task-description", cfg.language_instruction, "--config", args.config,
    ])

    # 4. LeRobot format (used by openpi's SFT trainer via --data.repo-id)
    run([
        "python", str(REPO_ROOT / "scripts" / "convert_maniskill_to_lerobot.py"),
        "--traj-path", str(rgb_h5), "--repo-name", cfg.lerobot_repo_id,
        "--task-description", cfg.language_instruction, "--config", args.config,
    ])
    # norm_stats intentionally NOT run here for *_joint_state SFT configs — they use
    # the DROID-official AssetsConfig baked into openpi's training/config.py, which
    # stage_sft's CLI never overrides. Only run --stage norm_stats separately if
    # get_sft_config_name(cfg) resolves to a config WITHOUT that baked override.


def stage_norm_stats(cfg, args):
    """Compute normalization statistics for the LeRobot dataset."""
    run([
        "uv", "run",
        str(OPENPI_SCRIPTS / "compute_norm_stats.py"),
        "--config-name", get_sft_config_name(cfg),
        "--repo-id", cfg.lerobot_repo_id,
    ])


def stage_sft(cfg, args, run_dir):
    """SFT warmup — fine-tune π₀.₅ on the demo dataset."""
    sft_output = os.path.join(run_dir, "sft")

    # num_data_sft (0 = use every episode) comes straight from the task YAML now —
    # auto-namespace the exp_name so a limited-demo run never collides with (or
    # overwrites) the full-dataset run's checkpoints.
    sft_exp_name = cfg.sft_exp_name
    if cfg.num_data_sft > 0:
        sft_exp_name = f"{cfg.sft_exp_name}_demos{cfg.num_data_sft}"

    cmd = [
        "uv", "run",
        str(OPENPI_SCRIPTS / "train.py"),
        get_sft_config_name(cfg),
        "--exp-name", sft_exp_name,
        "--data.repo-id", cfg.lerobot_repo_id,
        "--assets-base-dir", "./assets",
        "--checkpoint-base-dir", sft_output,
        f"--num-train-steps={cfg.sft_num_train_steps}",
        f"--batch-size={cfg.sft_batch_size}",
        f"--num-workers={cfg.sft_num_workers}",
        f"--save-interval={cfg.sft_save_interval}",
        f"--log-interval={cfg.sft_log_interval}",
        f"--project-name={cfg.project_name}",
    ]

    if cfg.num_data_sft > 0:
        cmd.append(f"--data.num-demos={cfg.num_data_sft}")

    if cfg.sft_resume:
        cmd.append("--resume")

    run(cmd)


def stage_rl(cfg, args, run_dir, resuming):
    """RL training with EXPOLearner.

    train_pi_robo.py loads the SAME task YAML itself (cfg = load_task_config(
    FLAGS.task_config)) and reads seed/max_steps/batch_size/checkpoint settings/
    output_dir/rl_resume_dir/num_data_rl/dataset paths/etc. directly from there
    — it does NOT expose these as separate CLI flags (unlike stage_sft's
    openpi train.py, which is tyro-based and does want everything via CLI).
    Only pass what train_pi_robo.py actually defines as CLI flags: --config,
    --task_config, --fsdp_devices, plus ml_collections --config.<field>=
    overrides (these work without an explicit flags.DEFINE, via
    config_flags.DEFINE_config_file). run_dir/resuming are accepted for
    signature consistency with the other stages but unused here —
    train_pi_robo.py resolves its own run directory from
    cfg.output_dir/cfg.rl_resume_dir independently.
    """
    cmd = [
        "python", str(TRAIN_PI_ROBO),
        "--config=configs/model/expo_ft_pi_config.py",
        f"--task_config={args.config}",
        f"--fsdp_devices={cfg.fsdp_devices}",
    ]

    if getattr(args, "sft_checkpoint", None) is not None:
        cmd.append(f"--config.pi05_weight_loader_path={Path(args.sft_checkpoint) / 'params'}")

    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["demos", "norm_stats", "sft", "rl", "all"], default="all")
    parser.add_argument(
        "--sft-checkpoint", default=None,
        help="Path to the SFT checkpoint directory to initialize RL from "
             "(e.g. logs/stack_cube/<run>/sft/<config_name>/<exp_name>/1400). "
             "Omit to use the config's default weight loader (base pretrained "
             "checkpoint, NOT SFT-finetuned — only appropriate if you deliberately "
             "want to skip SFT, e.g. for the SFT-warmup-necessity ablation).",
    )
    args = parser.parse_args()

    cfg = load_task_config(args.config)

    # resolve_run_dir() creates the directory immediately (os.makedirs) — only
    # call it when the requested stage actually writes there (sft/rl write
    # checkpoints under run_dir; demos writes to demos/, norm_stats doesn't
    # write under logs/ at all). Otherwise a demos-only or norm_stats-only run
    # left behind an empty, unused logs/<task>/<task>_expo_ft_<timestamp>/ dir.
    run_dir, resuming = (None, None)
    if args.stage in ("sft", "rl", "all"):
        run_dir, resuming = resolve_run_dir(cfg, resume_dir=cfg.sft_resume_dir)

    if args.stage in ("demos", "all"):
        stage_demos(cfg, args)

    if args.stage in ("norm_stats", "all"):
        stage_norm_stats(cfg, args)

    if args.stage in ("sft", "all"):
        stage_sft(cfg, args, run_dir)

    if args.stage in ("rl", "all"):
        stage_rl(cfg, args, run_dir, resuming)


if __name__ == "__main__":
    main()
