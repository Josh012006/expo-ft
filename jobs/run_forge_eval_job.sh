#!/bin/bash
# Usage:
#   sbatch jobs/run_forge_eval_job.sh [--checkpoint DIR] [--n-episodes N]
#     --checkpoint   SFT checkpoint STEP directory (the folder that contains params/). Omit: zero-shot pi05_droid_jointpos.
#     --n-episodes   number of episodes (default 5)
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#   sbatch jobs/run_forge_eval_job.sh                                        # zero-shot pi05_droid_jointpos, 5 episodes
#   sbatch jobs/run_forge_eval_job.sh --checkpoint <sft_checkpoint_step_dir> --n-episodes 20
#     e.g. sbatch jobs/run_forge_eval_job.sh --checkpoint logs/forge_sft/expo_pi05_droid_lora_finetune_sft_joint_state_delta/forge_A/4000 --n-episodes 20
#
#SBATCH --job-name=forge-eval
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --no-requeue

# Server + client, same node, same GPU, one allocation — the sbatch version
# of the two-tmux-window workflow. Defaults to a short test (5 episodes) so
# a mistake costs a few minutes, not an abandoned long-running job.
# An episode takes about 35 s: 20 episodes ~ 12 min, 50 episodes ~ 30 min (+ ~5 min start-up).
#
# Cancel-safety: `trap cleanup EXIT TERM INT` below fires on normal script
# completion AND on scancel (SLURM sends TERM, with a grace period before
# KILL) — so the server and its Isaac Sim subprocesses always get torn down,
# never left pending/orphaned on the node.

set -u  # catch unset-variable typos; NOT set -e (would fight the trap/background logic)

usage() {
    cat <<'EOF'
Usage: sbatch jobs/run_forge_eval_job.sh [--checkpoint DIR] [--n-episodes N]
  --checkpoint   SFT checkpoint STEP directory (the folder that contains params/); omit for zero-shot
  --n-episodes   number of episodes (default 5)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

CHECKPOINT=""              # optional: SFT checkpoint STEP directory (the folder that contains params/)
N_EPISODES=5
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) need_value "$1" "$#" "${2-}"; CHECKPOINT="$2"; shift 2 ;;
        --n-episodes) need_value "$1" "$#" "${2-}"; N_EPISODES="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

REPO_ROOT="$HOME/projects/expo-ft"
CONFIG="configs/task/isaaclab/peg_insert_forge_pi05.yaml"
SERVER_PORT=8102
SERVER_LOG="logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_server.log"

cd "$REPO_ROOT" || { echo "FATAL: cannot cd to $REPO_ROOT"; exit 1; }
mkdir -p logs/slurm || { echo "FATAL: cannot create logs/slurm"; exit 1; }

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping server (pid $SERVER_PID, process group)..."
        # Negative PID = signal the whole process group (setsid below put the
        # server and every Isaac Sim/Kit subprocess it spawns into one group
        # equal to this PID) — a plain `kill $SERVER_PID` would only hit the
        # top-level python process, not Kit's own children.
        kill -TERM -"$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 15); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[cleanup] server still up after 15s, SIGKILL."
            kill -KILL -"$SERVER_PID" 2>/dev/null
        fi
    fi
}
trap cleanup EXIT TERM INT

echo "[server] launching..."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(
    source .venv-isaaclab/bin/activate
    setsid python -m expo_ft.env.isaaclab.forge_server --config "$CONFIG"
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[server] pid=$SERVER_PID, log=$SERVER_LOG"

echo "[server] waiting for port $SERVER_PORT to accept connections..."
READY=0
for i in $(seq 1 60); do  # 60 x 5s = 5 min ceiling
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[server] process died before becoming ready — see $SERVER_LOG"
        exit 1
    fi
    if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
sys.exit(0 if s.connect_ex(('localhost', $SERVER_PORT)) == 0 else 1)
" 2>/dev/null; then
        READY=1
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "[server] never became ready within 5 min — see $SERVER_LOG"
    exit 1
fi
echo "[server] ready after $((i * 5))s."

echo "[client] running eval (checkpoint: ${CHECKPOINT:-none, base weights}, episodes: $N_EPISODES)..."
source .venv/bin/activate
CKPT_ARGS=()
if [ -n "$CHECKPOINT" ]; then
    if [ ! -d "$CHECKPOINT/params" ]; then
        echo "FATAL: $CHECKPOINT/params not found (pass the checkpoint STEP directory)"
        exit 1
    fi
    CKPT_ARGS=(--checkpoint "$CHECKPOINT")
fi
python scripts/eval_policy.py \
    --config "$CONFIG" \
    --n-episodes "$N_EPISODES" \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
    --video-dir "logs/eval_videos/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
CLIENT_STATUS=$?

echo "[client] exited with status $CLIENT_STATUS"
exit $CLIENT_STATUS
