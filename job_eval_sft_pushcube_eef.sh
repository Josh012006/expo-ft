#!/bin/bash
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

cd ~/projects/expo-ft
source scripts/setup_env.sh

python scripts/eval_policy.py \
    --config configs/task/maniskill/push_cube_eef.yaml \
    --checkpoint logs/push_cube/push_cube_expo_ft_2026-06-30_03-35-07/sft/expo_pi05_droid_lora_finetune_sft_cartesian_state/push_cube_sft/5000 \
    --n-episodes 200
