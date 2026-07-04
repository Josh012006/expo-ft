#!/bin/bash
# Usage:
#   sbatch job_sft.sh <venv_name> <config_path> [num_demos]
#
# Examples:
#   sbatch job_sft.sh .venv configs/task/maniskill/stack_cube.yaml
#       -> uses every episode in the LeRobot dataset
#   sbatch job_sft.sh .venv configs/task/maniskill/stack_cube.yaml 50
#       -> uses only the first 50 episodes, checkpoints land in a separate
#          "<sft_exp_name>_demos50" directory, does not touch the full-dataset run
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
CONFIG=${2:-configs/task/maniskill/stack_cube.yaml}
NUM_DEMOS=${3:-}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage sft \
    ${NUM_DEMOS:+--num-demos "$NUM_DEMOS"}
