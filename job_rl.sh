#!/bin/bash
# Usage:
#   sbatch job_rl.sh <venv_name> <config_path> [sft_checkpoint]
#
# num_data_rl (how many demo episodes for the offline replay buffer, 0 = all)
# now lives entirely in the task YAML — edit it there instead of passing it here.
#
# Examples:
#   sbatch job_rl.sh .venv configs/task/maniskill/stack_cube.yaml
#   sbatch job_rl.sh .venv configs/task/maniskill/stack_cube.yaml logs/stack_cube/stack_cube_expo_ft_2026-07-02_09-08-24/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft/2400
#
#SBATCH --job-name=expo_rl
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem-per-gpu=256G
#SBATCH --time=120:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/rl_%j.out
#SBATCH --no-requeue
VENV=${1:-.venv}
CONFIG=${2:-configs/task/maniskill/stack_cube_expo_ft.yaml}
SFT_CHECKPOINT=${3:-}
cd ~/projects/expo-ft
source scripts/setup_env.sh "$VENV"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
python3 scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage rl \
    ${SFT_CHECKPOINT:+--sft-checkpoint "$SFT_CHECKPOINT"}

# Chained AFTER the line above returns -- by then both run_pipeline.py and
# its train_pi_robo.py child have fully exited, releasing the GPU memory
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 held for their entire process lifetime.
# Running eval_curve.py while training's own process was still alive (just
# blocked waiting on it) starved every eval subprocess of GPU memory -- this
# is why the sweep is chained here at the shell level instead of called
# in-process from within train_pi_robo.py itself.
HANDOFF="logs/eval_curve_handoff_${SLURM_JOB_ID}.json"
if [ -f "$HANDOFF" ]; then
    python3 scripts/run_eval_curve_from_handoff.py --handoff "$HANDOFF"
fi
