#!/bin/bash
# Usage:
#   sbatch job_sft.sh <venv_name> <config_path>
#
# num_data_sft (how many demo episodes to use, 0 = all) now lives entirely in
# the task YAML — edit it there instead of passing it here.
#
#SBATCH --job-name=expo_sft
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=50:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/sft_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube_sft.yaml}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python3 scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage sft
