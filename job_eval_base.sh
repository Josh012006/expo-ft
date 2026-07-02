#!/bin/bash
# Usage:
#   sbatch job_eval_base.sh <venv_name> <config_path> <n_episodes>
#
# Examples:
#   sbatch job_eval_base.sh .venv configs/task/maniskill/stack_cube_eef.yaml 200
#   sbatch job_eval_base.sh .venv configs/task/maniskill/push_cube_eef.yaml 200
#   sbatch job_eval_base.sh .venv-robocasa configs/task/robocasa/close_drawer.yaml 200
#
#SBATCH --job-name=expo_eval
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=128G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/eval_%j.out
#SBATCH --no-requeue

VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube_eef.yaml}
N_EPISODES=${3:-200}

cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python scripts/eval_base_policy.py \
    --config "$CONFIG" \
    --n-episodes "$N_EPISODES"
