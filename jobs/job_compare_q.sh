#!/bin/bash
# Usage:
#   sbatch jobs/job_compare_q.sh --rl-checkpoints DIR[,DIR...] --reference-checkpoint DIR
#                                [--venv NAME] [--config PATH] [--n-states N]
#     --rl-checkpoints         (required) see below
#     --reference-checkpoint   (required) see below
#     --venv                   virtualenv to activate (default .venv)
#     --config                 task YAML (default configs/task/maniskill/push_cube_expo_ft.yaml)
#     --n-states               number of sampled states (default 100)
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# --rl-checkpoints: one or more RL/EXPOLearner checkpoint STEP directories,
#   comma-separated (e.g. .../checkpoints/20000,.../checkpoints/118000) --
#   the critic(s) being examined. Demo data, the reference checkpoint, and
#   the sampled batch of states are all loaded ONCE and reused across every
#   checkpoint listed, so passing several here is much cheaper than
#   separate submissions.
# --reference-checkpoint: path to the reference SFT checkpoint's STEP directory
#   (e.g. the 96% SR checkpoint, .../push_cube_sft_demos50/3200) -- a SEPARATE
#   π₀.₅, not the RL checkpoint's own frozen VLA. Same convention as
#   --rl-checkpoints above: pass the step dir, the "params" item inside it is
#   loaded automatically -- do NOT append /params yourself.
#
# Examples:
#   sbatch jobs/job_compare_q.sh --venv .venv --config configs/task/maniskill/push_cube_expo_ft.yaml \
#       --rl-checkpoints logs/push_cube/.../checkpoints/20000,logs/push_cube/.../checkpoints/118000 \
#       --reference-checkpoint logs/push_cube/.../sft/.../push_cube_sft_demos50/3200
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

usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_compare_q.sh --rl-checkpoints DIR[,DIR...] --reference-checkpoint DIR
                                    [--venv NAME] [--config PATH] [--n-states N]
  --rl-checkpoints        (required) RL/EXPOLearner checkpoint STEP directories, comma-separated if more than one
  --reference-checkpoint  (required) reference SFT checkpoint STEP directory (do NOT append /params)
  --venv                  virtualenv to activate (default .venv)
  --config                task YAML (default configs/task/maniskill/push_cube_expo_ft.yaml)
  --n-states              number of sampled states (default 100)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/maniskill/push_cube_expo_ft.yaml
RL_CHECKPOINTS=
REFERENCE_CHECKPOINT=
N_STATES=100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --rl-checkpoints) need_value "$1" "$#" "${2-}"; RL_CHECKPOINTS="$2"; shift 2 ;;
        --reference-checkpoint) need_value "$1" "$#" "${2-}"; REFERENCE_CHECKPOINT="$2"; shift 2 ;;
        --n-states) need_value "$1" "$#" "${2-}"; N_STATES="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
if [[ -z "$RL_CHECKPOINTS" ]]; then echo "ERROR: --rl-checkpoints is required (comma-separated if more than one)" >&2; usage >&2; exit 1; fi
if [[ -z "$REFERENCE_CHECKPOINT" ]]; then echo "ERROR: --reference-checkpoint is required" >&2; usage >&2; exit 1; fi

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1

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
