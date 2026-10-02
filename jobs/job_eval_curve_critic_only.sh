#!/bin/bash
# Usage:
#   sbatch job_eval_curve_critic_only.sh <venv_name> <config_path> <checkpoints_dir> <n_episodes> [start_checkpoint]
#
# checkpoints_dir: an RL run's own checkpoints/ directory (numeric step
# subfolders) -- e.g. logs/stack_cube/<run>_rl/checkpoints. SFT-only
# checkpoints don't have a trained critic, so this job only makes sense for
# RL/EXPOLearner checkpoints (same requirement as job_eval_curve.sh's
# rl_curve mode).
#
# start_checkpoint: the SFT checkpoint the RL run actually started from --
# evaluated once as the step=0 reference point on the "full" curve (see
# eval_curve_critic_only.py's --start-checkpoint help). Optional.
#
# Compares, at every checkpoint: the full pipeline (residual + critic
# selection) vs. critic-only (n_edit_samples=0 -- critic still picks among
# raw VLA samples, residual never invoked). Same fixed episode seeds for
# both, per checkpoint.
#
# Example:
#   sbatch job_eval_curve_critic_only.sh .venv configs/task/maniskill/stack_cube.yaml \
#       logs/stack_cube/stack_cube_expo_ft_2026-07-05_21-40-48_rl/checkpoints \
#       200 \
#       logs/stack_cube/stack_cube_expo_ft_2026-07-05_01-06-12/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft_demos50/3999
#
# Safe to re-run/resubmit: already-evaluated checkpoints are skipped (see
# --force in eval_curve_critic_only.py if you actually want to redo them).
#
#SBATCH --job-name=expo_eval_curve_critic_only
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/eval_curve_critic_only_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube.yaml}
CHECKPOINTS_DIR=${3}
N_EPISODES=${4:-200}
START_CHECKPOINT=${5:-}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python3 scripts/eval_curve_critic_only.py \
    --config "$CONFIG" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --n-episodes "$N_EPISODES" \
    ${START_CHECKPOINT:+--start-checkpoint "$START_CHECKPOINT"}
