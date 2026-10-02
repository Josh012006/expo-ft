#!/bin/bash
# Usage:
#   sbatch job_compare_q.sh <venv_name> <config_path> <rl_checkpoints> <reference_checkpoint> [n_states]
#
# rl_checkpoints: one or more RL/EXPOLearner checkpoint STEP directories,
#   comma-separated (e.g. .../checkpoints/20000,.../checkpoints/118000) --
#   the critic(s) being examined. Demo data, the reference checkpoint, and
#   the sampled batch of states are all loaded ONCE and reused across every
#   checkpoint listed, so passing several here is much cheaper than
#   separate submissions.
# reference_checkpoint: path to the reference SFT checkpoint's STEP directory
#   (e.g. the 96% SR checkpoint, .../push_cube_sft_demos50/3200) -- a SEPARATE
#   π₀.₅, not the RL checkpoint's own frozen VLA. Same convention as
#   rl_checkpoints below: pass the step dir, the "params" item inside it is
#   loaded automatically -- do NOT append /params yourself.
#
# Examples:
#   sbatch job_compare_q.sh .venv configs/task/maniskill/push_cube_expo_ft.yaml \
#       logs/push_cube/.../checkpoints/20000,logs/push_cube/.../checkpoints/118000 \
#       logs/push_cube/.../sft/.../push_cube_sft_demos50/3200
#
#SBATCH --job-name=expo_compare_q
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=120G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/compare_%j.out
#SBATCH --no-requeue

VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/push_cube_expo_ft.yaml}
RL_CHECKPOINTS=${3:?rl_checkpoints is required (comma-separated if more than one)}
REFERENCE_CHECKPOINT=${4:?reference_checkpoint is required}
N_STATES=${5:-100}

cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"

# Loads two full π₀.₅ instances at once (RL checkpoint's frozen VLA +
# separate reference checkpoint) -- untested territory memory-wise, every
# other job so far only ever loads one. Reserve as much of the GPU upfront
# as possible to reduce fragmentation-related OOM risk.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Build one --rl-checkpoint flag per comma-separated entry.
IFS=',' read -ra CKPT_ARRAY <<< "$RL_CHECKPOINTS"
CKPT_FLAGS=()
for ckpt in "${CKPT_ARRAY[@]}"; do
    CKPT_FLAGS+=(--rl-checkpoint "$ckpt")
done

python3 scripts/compare_argmax_vs_reference_q.py \
    --config "$CONFIG" \
    "${CKPT_FLAGS[@]}" \
    --reference-checkpoint "$REFERENCE_CHECKPOINT" \
    --n-states "$N_STATES" \
    --output-json "logs/q_comparison_${SLURM_JOB_ID}.json" \
    --output-csv "logs/q_comparison_${SLURM_JOB_ID}.csv"
