#!/bin/bash
# Usage:
#   sbatch jobs/job_demos.sh [--venv NAME] [--config PATH]
#     --venv    virtualenv to activate (default .venv)
#     --config  task YAML (default configs/task/maniskill/stack_cube.yaml)
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# Examples:
#   sbatch jobs/job_demos.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml
#   sbatch jobs/job_demos.sh --venv .venv --config configs/task/maniskill/pick_cube.yaml
#   sbatch jobs/job_demos.sh --venv .venv --config configs/task/maniskill/push_cube.yaml
#
#SBATCH --job-name=expo_demos
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/demos_%j.out
#SBATCH --no-requeue
usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_demos.sh [--venv NAME] [--config PATH]
  --venv    virtualenv to activate (default .venv)
  --config  task YAML (default configs/task/maniskill/stack_cube.yaml)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/maniskill/stack_cube.yaml
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1
python3 scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage demos
