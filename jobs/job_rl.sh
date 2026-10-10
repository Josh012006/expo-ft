#!/bin/bash
# Usage:
#   sbatch jobs/job_rl.sh [--venv NAME] [--config PATH] [--sft-checkpoint DIR]
#     --venv            virtualenv to activate (default .venv)
#     --config          task YAML (default configs/task/maniskill/stack_cube_expo_ft.yaml)
#     --sft-checkpoint  SFT checkpoint STEP directory the RL run starts from (default: the config's base weights)
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# num_data_rl (how many demo episodes for the offline replay buffer, 0 = all)
# now lives entirely in the task YAML — edit it there instead of passing it here.
#
# Examples:
#   sbatch jobs/job_rl.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml
#   sbatch jobs/job_rl.sh --venv .venv --config configs/task/maniskill/stack_cube.yaml \
#       --sft-checkpoint logs/stack_cube/stack_cube_expo_ft_2026-07-02_09-08-24/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft/2400
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
usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_rl.sh [--venv NAME] [--config PATH] [--sft-checkpoint DIR]
  --venv            virtualenv to activate (default .venv)
  --config          task YAML (default configs/task/maniskill/stack_cube_expo_ft.yaml)
  --sft-checkpoint  SFT checkpoint STEP directory the RL run starts from (default: the config's base weights)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/maniskill/stack_cube_expo_ft.yaml
SFT_CHECKPOINT=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --sft-checkpoint) need_value "$1" "$#" "${2-}"; SFT_CHECKPOINT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

cd ~/projects/expo-ft
source scripts/setup_env.sh --venv "$VENV" || exit 1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
python3 scripts/run_pipeline.py \
    --config "$CONFIG" \
    --stage rl \
    ${SFT_CHECKPOINT:+--sft-checkpoint "$SFT_CHECKPOINT"}

# NOTE: this used to chain scripts/run_eval_curve_from_handoff.py here
# automatically. Removed -- eval/success_rate is now primed on the starting
# model before any weight updates (see train_pi_robo.py), so it's
# trustworthy throughout the run and an automatic 200-seed sweep after
# every job is no longer needed by default. Run scripts/eval_curve.py
# directly (see job_eval_curve.sh) whenever a rigorous fixed-seed sweep is
# specifically wanted.
