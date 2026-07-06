#!/bin/bash
# Usage:
#   sbatch job_eval_curve.sh <venv_name> <config_path> <checkpoints_dir> <n_episodes> [save_videos] [start_checkpoint]
#
# start_checkpoint: for an RL checkpoints_dir, pass the SFT checkpoint the RL
# run actually started from — used as the 'base' reference point on the curve
# instead of the raw pretrained model (which wouldn't reflect what RL
# improved upon). Omit for an SFT checkpoints_dir, where the true base
# pretrained model IS the right reference point.
#
# Example:
#   sbatch job_eval_curve.sh .venv configs/task/maniskill/stack_cube.yaml \
#       logs/stack_cube/stack_cube_expo_ft_2026-07-05_21-40-48_rl/checkpoints \
#       200 save_videos \
#       logs/stack_cube/stack_cube_expo_ft_2026-07-05_01-06-12/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft_demos50/3999
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
START_CHECKPOINT=${6:-}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python scripts/eval_curve.py \
    --config "$CONFIG" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --n-episodes "$N_EPISODES" \
    ${SAVE_VIDEOS:+--save-videos} \
    ${START_CHECKPOINT:+--start-checkpoint "$START_CHECKPOINT"}
