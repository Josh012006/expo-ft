#!/bin/bash
# Usage:
#   sbatch jobs/job_eval.sh [--venv NAME] [--config PATH] [--n-episodes N] [--checkpoint DIR] [--rl-checkpoint DIR]
#     --venv           virtualenv to activate (default .venv)
#     --config         task YAML (default configs/task/maniskill/stack_cube.yaml)
#     --n-episodes     number of evaluation episodes (default 200)
#     --checkpoint     SFT/openpi-style checkpoint path — evaluates the frozen VLA only.
#     --rl-checkpoint  RL/EXPOLearner checkpoint STEP directory (e.g.
#                      .../checkpoints/40000) — loads the full trained agent (VLA + residual
#                      policy + critic) and evaluates with only_base_actions=False.
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# Pass only one of --checkpoint / --rl-checkpoint. If both are given, --rl-checkpoint wins.
#
# Examples:
#   sbatch jobs/job_eval.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml --n-episodes 200
#   sbatch jobs/job_eval.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml --n-episodes 200 \
#       --checkpoint logs/stack_cube/.../sft/.../3999
#   sbatch jobs/job_eval.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml --n-episodes 200 \
#       --rl-checkpoint logs/stack_cube/.../checkpoints/40000
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

usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_eval.sh [--venv NAME] [--config PATH] [--n-episodes N] [--checkpoint DIR] [--rl-checkpoint DIR]
  --venv           virtualenv to activate (default .venv)
  --config         task YAML (default configs/task/maniskill/stack_cube.yaml)
  --n-episodes     number of evaluation episodes (default 200)
  --checkpoint     SFT/openpi-style checkpoint path (frozen VLA only)
  --rl-checkpoint  RL/EXPOLearner checkpoint STEP directory (full trained agent); wins over --checkpoint
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/maniskill/stack_cube.yaml
N_EPISODES=200
CHECKPOINT=
RL_CHECKPOINT=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --n-episodes) need_value "$1" "$#" "${2-}"; N_EPISODES="$2"; shift 2 ;;
        --checkpoint) need_value "$1" "$#" "${2-}"; CHECKPOINT="$2"; shift 2 ;;
        --rl-checkpoint) need_value "$1" "$#" "${2-}"; RL_CHECKPOINT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1
python3 scripts/eval_policy.py \
    --config "$CONFIG" \
    --n-episodes "$N_EPISODES" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${RL_CHECKPOINT:+--rl-checkpoint "$RL_CHECKPOINT"}
