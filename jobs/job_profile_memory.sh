#!/bin/bash
# Usage:
#   sbatch jobs/job_profile_memory.sh --config PATH [--venv NAME] [--checkpoint DIR] [--num-demos N] [--n-updates N]
#                                     [--mem-fraction F] [--skip-update]
#     --config         task YAML of the RL run to size (required)
#     --venv           virtualenv to activate (default .venv)
#     --checkpoint     SFT checkpoint STEP directory (default: base pi0.5 weights; same memory)
#     --num-demos      demonstration episodes loaded (default 5; the GPU need does not depend on it)
#     --n-updates      timed steady-state agent.update calls (default 3)
#     --mem-fraction   cap JAX at this fraction of the card (XLA_PYTHON_CLIENT_MEM_FRACTION): "does it fit in 48 GB?"
#                      on an 80 GB card = 0.6. An OOM then IS the answer; the stages already printed stay valid.
#     --skip-update    inference numbers only (stops after sample_actions)
#   (sbatch's own options go BEFORE the script name, e.g. to size on another card:
#    sbatch --gres=gpu:l40s:1 jobs/job_profile_memory.sh --config ...)
#
# Measures GPU / host memory and seconds per call of the real RL programs on demonstration data, with no environment and
# no training (scripts/profile_memory.py). ~10 minutes, most of it compilation. Output:
#   logs/profile_memory_<jobid>.out                 the stage table + the final "size the GPU with" lines
#   logs/profile_memory/<config>_<time>.json        the same, machine-readable
#   logs/profile_memory/nvidia_smi_<jobid>.csv      GPU memory of every process on the card, every 5 s, whole job
#
# Examples:
#   sbatch jobs/job_profile_memory.sh --config configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml
#   sbatch jobs/job_profile_memory.sh --config configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml --mem-fraction 0.6
#
#SBATCH --job-name=expo_profile_mem
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=128G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/profile_memory_%j.out
#SBATCH --no-requeue

usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_profile_memory.sh --config PATH [--venv NAME] [--checkpoint DIR] [--num-demos N] [--n-updates N]
                                         [--mem-fraction F] [--skip-update]
  --config         task YAML of the RL run to size (required)
  --venv           virtualenv to activate (default .venv)
  --checkpoint     SFT checkpoint STEP directory (default: base pi0.5 weights)
  --num-demos      demonstration episodes loaded (default 5)
  --n-updates      timed steady-state agent.update calls (default 3)
  --mem-fraction   cap JAX at this fraction of the card (0.6 on an 80 GB card ~ 48 GB)
  --skip-update    inference numbers only
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=
CHECKPOINT=
NUM_DEMOS=5
N_UPDATES=3
MEM_FRACTION=
SKIP_UPDATE=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --checkpoint) need_value "$1" "$#" "${2-}"; CHECKPOINT="$2"; shift 2 ;;
        --num-demos) need_value "$1" "$#" "${2-}"; NUM_DEMOS="$2"; shift 2 ;;
        --n-updates) need_value "$1" "$#" "${2-}"; N_UPDATES="$2"; shift 2 ;;
        --mem-fraction) need_value "$1" "$#" "${2-}"; MEM_FRACTION="$2"; shift 2 ;;
        --skip-update) SKIP_UPDATE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
[[ -n "$CONFIG" ]] || { echo "ERROR: --config is required" >&2; usage >&2; exit 1; }

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1

mkdir -p logs/profile_memory
SMI_LOG="logs/profile_memory/nvidia_smi_${SLURM_JOB_ID:-local}.csv"
# GPU memory per process on the card, whole job: the timeline behind the stage table (and, in a job that also runs an
# Isaac Lab server, the split between the server and the JAX client: the process_name column tells them apart).
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv -l 5 > "$SMI_LOG" 2>&1 &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null' EXIT TERM INT

# No preallocation (scripts/profile_memory.py sets it too): the point is the need, not a reservation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ -n "$MEM_FRACTION" ]]; then export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"; fi

python3 scripts/profile_memory.py \
    --config "$CONFIG" \
    --num-demos "$NUM_DEMOS" \
    --n-updates "$N_UPDATES" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${SKIP_UPDATE:+--skip-update}
STATUS=$?
echo "profile_memory.py exit status: $STATUS (nvidia-smi timeline: $SMI_LOG)"
exit $STATUS
