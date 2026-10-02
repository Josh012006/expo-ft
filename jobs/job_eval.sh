#!/bin/bash
# Usage:
#   sbatch job_eval.sh <venv_name> <config_path> <n_episodes> [checkpoint] [rl_checkpoint]
#
# checkpoint: SFT/openpi-style checkpoint path — evaluates the frozen VLA only.
# rl_checkpoint: RL/EXPOLearner checkpoint STEP directory (e.g.
#   .../checkpoints/40000) — loads the full trained agent (VLA + residual
#   policy + critic) and evaluates with only_base_actions=False.
#
# Pass only one of the two. If both are given, rl_checkpoint wins.
#
# Examples:
#   sbatch job_eval.sh .venv configs/task/maniskill/stack_cube.yaml 200
#   sbatch job_eval.sh .venv configs/task/maniskill/stack_cube.yaml 200 \
#       logs/stack_cube/.../sft/.../3999
#   sbatch job_eval.sh .venv configs/task/maniskill/stack_cube.yaml 200 "" \
#       logs/stack_cube/.../checkpoints/40000
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
RL_CHECKPOINT=${5:-}

cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
python3 scripts/eval_policy.py \
    --config "$CONFIG" \
    --n-episodes "$N_EPISODES" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${RL_CHECKPOINT:+--rl-checkpoint "$RL_CHECKPOINT"}
