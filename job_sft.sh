#!/bin/bash
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

cd ~/projects/expo-ft
source scripts/setup_env.sh

python scripts/run_pipeline.py \
    --config configs/task/maniskill_stack_cube.yaml \
    --stage sft
