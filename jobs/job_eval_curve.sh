#!/bin/bash
# Usage:
#   sbatch jobs/job_eval_curve.sh --checkpoints-dir DIR [--venv NAME] [--config PATH] [--n-episodes N]
#                                 [--save-videos] [--start-checkpoint DIR] [--rl-curve]
#     --checkpoints-dir   (required) folder of numeric step sub-folders to evaluate
#     --venv              virtualenv to activate (default .venv)
#     --config            task YAML (default configs/task/maniskill/stack_cube.yaml)
#     --n-episodes        episodes per checkpoint (default 50)
#     --save-videos       switch: record videos
#     --start-checkpoint  see below
#     --rl-curve          switch: see below
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# --rl-curve: pass it when
# --checkpoints-dir contains RL/EXPOLearner checkpoints (not SFT ones) — this
# loads the full trained agent (residual policy + critic included) instead
# of silently evaluating just the frozen VLA. Required for any RL curve.
#
# --start-checkpoint: for an RL --checkpoints-dir, pass the SFT checkpoint the RL
# run actually started from — used as the 'base' reference point on the curve
# instead of the raw pretrained model (which wouldn't reflect what RL
# improved upon). Omit for an SFT checkpoints_dir, where the true base
# pretrained model IS the right reference point.
#
# Example (RL curve):
#   sbatch jobs/job_eval_curve.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml \
#       --checkpoints-dir logs/stack_cube/stack_cube_expo_ft_2026-07-05_21-40-48_rl/checkpoints \
#       --n-episodes 200 --save-videos \
#       --start-checkpoint logs/stack_cube/stack_cube_expo_ft_2026-07-05_01-06-12/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft_demos50/3999 \
#       --rl-curve
#
# Safe to re-run/resubmit: already-evaluated checkpoints are skipped (see --force
# in eval_curve.py if you actually want to redo them).
#
#SBATCH --job-name=expo_eval_curve
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/eval_curve_%j.out
#SBATCH --no-requeue
usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_eval_curve.sh --checkpoints-dir DIR [--venv NAME] [--config PATH] [--n-episodes N]
                                     [--save-videos] [--start-checkpoint DIR] [--rl-curve]
  --checkpoints-dir   (required) folder of numeric step sub-folders to evaluate
  --venv              virtualenv to activate (default .venv)
  --config            task YAML (default configs/task/maniskill/stack_cube.yaml)
  --n-episodes        episodes per checkpoint (default 50)
  --save-videos       switch: record videos
  --start-checkpoint  SFT checkpoint the RL run started from (reference point of an RL curve)
  --rl-curve          switch: the folder holds RL/EXPOLearner checkpoints (loads the full agent)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/maniskill/stack_cube.yaml
CHECKPOINTS_DIR=
N_EPISODES=50
SAVE_VIDEOS=
START_CHECKPOINT=
RL_CURVE=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --checkpoints-dir) need_value "$1" "$#" "${2-}"; CHECKPOINTS_DIR="$2"; shift 2 ;;
        --n-episodes) need_value "$1" "$#" "${2-}"; N_EPISODES="$2"; shift 2 ;;
        --save-videos) SAVE_VIDEOS=1; shift ;;
        --start-checkpoint) need_value "$1" "$#" "${2-}"; START_CHECKPOINT="$2"; shift 2 ;;
        --rl-curve) RL_CURVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
if [[ -z "$CHECKPOINTS_DIR" ]]; then echo "ERROR: --checkpoints-dir is required" >&2; usage >&2; exit 1; fi

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1
python3 scripts/eval_curve.py \
    --config "$CONFIG" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --n-episodes "$N_EPISODES" \
    ${SAVE_VIDEOS:+--save-videos} \
    ${START_CHECKPOINT:+--start-checkpoint "$START_CHECKPOINT"} \
    ${RL_CURVE:+--rl-curve}
