#!/bin/bash
# Usage:
#   sbatch job_eval.sh <venv_name> <config_path> <n_episodes> [checkpoint]
#
# Examples:
#   sbatch job_eval.sh .venv configs/task/maniskill/stack_cube.yaml 200
#   sbatch job_eval.sh .venv configs/task/maniskill/push_cube.yaml 200
#   sbatch job_eval.sh .venv-robocasa configs/task/robocasa/close_drawer.yaml 200
#
#SBATCH --job-name=expo_eval
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/eval_%j.out
#SBATCH --no-requeue

VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube.yaml}
N_EPISODES=${3:-200}
CHECKPOINT=${4:-}

cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python scripts/eval_policy.py \
    --config "$CONFIG" \
    --n-episodes "$N_EPISODES" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}
