#!/bin/bash
#SBATCH --job-name=expo_sft
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=128G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/sft_%j.out
#SBATCH --no-requeue

cd ~/projects/expo-ft
source scripts/setup_env.sh

python scripts/run_pipeline.py \
    --config configs/task/stack_cube.yaml \
    --stage sft
