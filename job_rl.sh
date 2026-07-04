#!/bin/bash
# Usage:
#   sbatch job_rl.sh <venv_name> <config_path> [num_demos] [sft_checkpoint]
#
# Examples:
#   sbatch job_rl.sh .venv configs/task/maniskill/stack_cube.yaml
#   sbatch job_rl.sh .venv configs/task/maniskill/push_cube.yaml 50
#   sbatch job_rl.sh .venv configs/task/maniskill/stack_cube.yaml "" logs/stack_cube/stack_cube_expo_ft_2026-07-02_09-08-24/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft/2400
#
#SBATCH --job-name=expo_rl
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=120:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/rl_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube.yaml}
NUM_DEMOS=${3:-}
SFT_CHECKPOINT=${4:-}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage rl \
    ${NUM_DEMOS:+--num-demos "$NUM_DEMOS"} \
    ${SFT_CHECKPOINT:+--sft-checkpoint "$SFT_CHECKPOINT"}
