#!/bin/bash
# Usage:
#   sbatch jobs/job_eval_curve_critic_only.sh --checkpoints-dir DIR [--venv NAME] [--config PATH]
#                                             [--n-episodes N] [--start-checkpoint DIR]
#     --checkpoints-dir   (required) see below
#     --venv              virtualenv to activate (default .venv)
#     --config            task YAML (default configs/task/maniskill/stack_cube.yaml)
#     --n-episodes        episodes per checkpoint (default 200)
#     --start-checkpoint  see below
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# --checkpoints-dir: an RL run's own checkpoints/ directory (numeric step
# subfolders) -- e.g. logs/stack_cube/<run>_rl/checkpoints. SFT-only
# checkpoints don't have a trained critic, so this job only makes sense for
# RL/EXPOLearner checkpoints (same requirement as job_eval_curve.sh's
# --rl-curve mode).
#
# --start-checkpoint: the SFT checkpoint the RL run actually started from --
# evaluated once as the step=0 reference point on the "full" curve (see
# eval_curve_critic_only.py's --start-checkpoint help). Optional.
#
# Compares, at every checkpoint: the full pipeline (residual + critic
# selection) vs. critic-only (n_edit_samples=0 -- critic still picks among
# raw VLA samples, residual never invoked). Same fixed episode seeds for
# both, per checkpoint.
#
# Example:
#   sbatch jobs/job_eval_curve_critic_only.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml \
#       --checkpoints-dir logs/stack_cube/stack_cube_expo_ft_2026-07-05_21-40-48_rl/checkpoints \
#       --n-episodes 200 \
#       --start-checkpoint logs/stack_cube/stack_cube_expo_ft_2026-07-05_01-06-12/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft_demos50/3999
#
# Safe to re-run/resubmit: already-evaluated checkpoints are skipped (see
# --force in eval_curve_critic_only.py if you actually want to redo them).
#
#SBATCH --job-name=expo_eval_curve_critic_only
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/eval_curve_critic_only_%j.out
#SBATCH --no-requeue
usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_eval_curve_critic_only.sh --checkpoints-dir DIR [--venv NAME] [--config PATH]
                                                 [--n-episodes N] [--start-checkpoint DIR]
  --checkpoints-dir   (required) an RL run's checkpoints/ directory (numeric step sub-folders)
  --venv              virtualenv to activate (default .venv)
  --config            task YAML (default configs/task/maniskill/stack_cube.yaml)
  --n-episodes        episodes per checkpoint (default 200)
  --start-checkpoint  SFT checkpoint the RL run started from (step 0 reference point, optional)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/maniskill/stack_cube.yaml
CHECKPOINTS_DIR=
N_EPISODES=200
START_CHECKPOINT=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --checkpoints-dir) need_value "$1" "$#" "${2-}"; CHECKPOINTS_DIR="$2"; shift 2 ;;
        --n-episodes) need_value "$1" "$#" "${2-}"; N_EPISODES="$2"; shift 2 ;;
        --start-checkpoint) need_value "$1" "$#" "${2-}"; START_CHECKPOINT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
if [[ -z "$CHECKPOINTS_DIR" ]]; then echo "ERROR: --checkpoints-dir is required" >&2; usage >&2; exit 1; fi

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1
python3 scripts/eval_curve_critic_only.py \
    --config "$CONFIG" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --n-episodes "$N_EPISODES" \
    ${START_CHECKPOINT:+--start-checkpoint "$START_CHECKPOINT"}
