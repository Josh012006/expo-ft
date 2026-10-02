#!/bin/bash
# Usage:
#   sbatch job_demos.sh <venv_name> <config_path>
#
# Examples:
#   sbatch job_demos.sh .venv configs/task/maniskill/stack_cube.yaml
#   sbatch job_demos.sh .venv configs/task/maniskill/pick_cube.yaml
#   sbatch job_demos.sh .venv configs/task/maniskill/push_cube.yaml
#
#SBATCH --job-name=expo_demos
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/demos_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube.yaml}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python3 scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage demos
