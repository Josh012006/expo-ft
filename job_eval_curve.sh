#!/bin/bash
# Usage:
#   sbatch job_eval_curve.sh <venv_name> <config_path> <checkpoints_dir> <n_episodes> [save_videos]
#
# Pass "save_videos" (any non-empty value) as the 5th arg to save a rollout
# video per episode for every checkpoint.
#
# Example:
#   sbatch job_eval_curve.sh .venv configs/task/maniskill/stack_cube.yaml \
#       logs/stack_cube/stack_cube_expo_ft_2026-07-02_09-08-24/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft \
#       50
#   sbatch job_eval_curve.sh .venv configs/task/maniskill/stack_cube.yaml \
#       logs/stack_cube/.../checkpoints 200 save_videos
#
# Safe to re-run/resubmit: already-evaluated checkpoints are skipped (see --force
# in eval_curve.py if you actually want to redo them).
#
#SBATCH --job-name=expo_eval_curve
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/eval_curve_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube.yaml}
CHECKPOINTS_DIR=${3}
N_EPISODES=${4:-50}
SAVE_VIDEOS=${5:-}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python scripts/eval_curve.py \
    --config "$CONFIG" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --n-episodes "$N_EPISODES" \
    ${SAVE_VIDEOS:+--save-videos}
