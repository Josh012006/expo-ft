#!/bin/bash
# SFT (LoRA) of pi05_droid_jointpos on the FORGE waypoint demos. Same pipeline as the ManiSkill SFT (job_sft.sh):
# everything (steps, batch size, save interval, dataset, openpi config...) is read from the SFT YAML.
#
# Usage:
#   sbatch jobs/job_forge_sft.sh [--venv NAME] [--config PATH] [--output-dir DIR]
#     --venv        virtualenv to activate (default .venv)
#     --config      SFT YAML (default configs/task/isaaclab/peg_insert_forge_sft.yaml)
#     --output-dir  where the run folder is created (default: output_dir of the YAML, logs/forge_peg_insert).
#                   Checkpoints: <output-dir>/<run_name>_<date>-<jobid>/sft/<openpi config>/<exp_name>/<step>/
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#   e.g. sbatch jobs/job_forge_sft.sh --venv .venv --config configs/task/isaaclab/peg_insert_forge_sft.yaml --output-dir logs/forge_peg_insert
#
# Prerequisite: the LeRobot dataset <lerobot_home>/<lerobot_repo_id> of the YAML exists (conversion script).
#
#SBATCH --job-name=forge_sft
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=50:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/forge_sft_%j.out
#SBATCH --no-requeue
usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_forge_sft.sh [--venv NAME] [--config PATH] [--output-dir DIR]
  --venv        virtualenv to activate (default .venv)
  --config      SFT YAML (default configs/task/isaaclab/peg_insert_forge_sft.yaml)
  --output-dir  where the run folder is created (default: output_dir of the YAML)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/isaaclab/peg_insert_forge_sft.yaml
OUTPUT_DIR=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --output-dir) need_value "$1" "$#" "${2-}"; OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

cd ~/projects/expo-ft || exit 1
source scripts/setup_env.sh --venv "$VENV" || exit 1
mkdir -p logs

ARGS=(--config "$CONFIG" --stage sft)
if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi
python3 scripts/run_pipeline.py "${ARGS[@]}"
