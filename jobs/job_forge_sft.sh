#!/bin/bash
# SFT (LoRA) of pi05_droid_jointpos on the FORGE waypoint demos.
#
# Usage:
#   sbatch jobs/job_forge_sft.sh [exp_name] [num_train_steps] [venv]
#   e.g. sbatch jobs/job_forge_sft.sh forge_A 4001
#
# Prerequisite: the LeRobot dataset <LEROBOT_HOME>/<REPO_NAME> exists, i.e. the --output-dir and --repo-name given to
#   scripts/isaaclab/convert_waypoint_demos_to_lerobot.py (defaults here: demos/isaaclab/lerobot, expo_ft/forge_peg_insert).
#   Another folder: LEROBOT_HOME=path sbatch jobs/job_forge_sft.sh ...
#
# Normalization "A": the config below reloads the official DROID norm_stats of pi05_droid_jointpos (baked in
# the openpi config, like the ManiSkill joint-state SFT). Nothing is recomputed.
# The 7 joint actions (absolute targets in the dataset) become offsets from the chunk-start state at training
# time (DeltaActions); the gripper stays absolute.
#
# Checkpoints: logs/forge_sft/<config>/<exp_name>/<step>/   (step 4000 is saved when num_train_steps = 4001;
# --keep-period 2000 keeps steps 2000 and 4000, the others are deleted along the way).
# exp_name must be new: openpi refuses an existing one (add --resume or --overwrite below if you really mean it).
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

EXP_NAME=${1:-forge_A}
STEPS=${2:-4001}
VENV=${3:-.venv}
REPO_NAME=${REPO_NAME:-expo_ft/forge_peg_insert}
LEROBOT_HOME=${LEROBOT_HOME:-demos/isaaclab/lerobot}
CONFIG_NAME=expo_pi05_droid_lora_finetune_sft_joint_state_delta

cd ~/projects/expo-ft || exit 1
source scripts/setup_env.sh "$VENV"     # activates the venv (it also sets HF_LEROBOT_HOME=demos/lerobot: overridden below)
case "$LEROBOT_HOME" in /*) ;; *) LEROBOT_HOME="$PWD/$LEROBOT_HOME" ;; esac
export HF_LEROBOT_HOME="$LEROBOT_HOME"
mkdir -p logs

if [ ! -d "$HF_LEROBOT_HOME/$REPO_NAME" ]; then
    echo "FATAL: dataset $HF_LEROBOT_HOME/$REPO_NAME not found. Run convert_waypoint_demos_to_lerobot.py first."
    exit 1
fi

# Same recipe as the ManiSkill SFT (LoRA, batch 64, lr 2.5e-5 cosine from the config).
uv run expo_ft/agents/vla/openpi/scripts/train.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --data.repo-id "$REPO_NAME" \
    --assets-base-dir ./assets \
    --checkpoint-base-dir logs/forge_sft \
    --num-train-steps "$STEPS" \
    --batch-size 64 \
    --num-workers 12 \
    --save-interval 1000 \
    --keep-period 2000 \
    --log-interval 50 \
    --project-name expo-ft \
    --fsdp-devices 1
