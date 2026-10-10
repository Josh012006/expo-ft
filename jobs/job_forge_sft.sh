#!/bin/bash
# SFT (LoRA) of pi05_droid_jointpos on the FORGE waypoint demos. Same pipeline as the ManiSkill SFT (job_sft.sh):
# everything (steps, batch size, save interval, dataset, openpi config...) is read from the SFT YAML.
#
# Usage:
#   sbatch jobs/job_forge_sft.sh [venv] [config] [output_dir]
#     venv        default .venv
#     config      default configs/task/isaaclab/peg_insert_forge_sft.yaml
#     output_dir  where the run folder is created (default: output_dir of the YAML, logs/forge_peg_insert).
#                 Checkpoints: <output_dir>/<run_name>_<date>-<jobid>/sft/<openpi config>/<exp_name>/<step>/
#   e.g. sbatch jobs/job_forge_sft.sh .venv configs/task/isaaclab/peg_insert_forge_sft.yaml logs/forge_peg_insert
#
# Prerequisite: the LeRobot dataset <lerobot_home>/<lerobot_repo_id> of the YAML exists (conversion script).
#
#SBATCH --job-name=forge_sft
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=08:00:00
#SBATCH --exclude=cn-g001,cn-g007,cn-g008,cn-g010,cn-g011,cn-g012,cn-g013,cn-g014,cn-g015,cn-g017,cn-g018,cn-g024,cn-g025,cn-g026,cn-d003,cn-i001
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/forge_sft_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/isaaclab/peg_insert_forge_sft.yaml}
OUTPUT_DIR=${3:-}
cd ~/projects/expo-ft || exit 1
source scripts/setup_env.sh "$VENV"
mkdir -p logs

ARGS=(--config "$CONFIG" --stage sft)
if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi
python3 scripts/run_pipeline.py "${ARGS[@]}"
