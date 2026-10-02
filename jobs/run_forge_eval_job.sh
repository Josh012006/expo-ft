#!/bin/bash
#SBATCH --job-name=forge-eval-test
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --no-requeue

# Server + client, same node, same GPU, one allocation — the sbatch version
# of the two-tmux-window workflow. First test: short --time, 5 episodes, so
# a mistake costs a few minutes, not an abandoned long-running job.
#
# Cancel-safety: `trap cleanup EXIT TERM INT` below fires on normal script
# completion AND on scancel (SLURM sends TERM, with a grace period before
# KILL) — so the server and its Isaac Sim subprocesses always get torn down,
# never left pending/orphaned on the node.

set -u  # catch unset-variable typos; NOT set -e (would fight the trap/background logic)

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

echo "[client] running eval..."
source .venv/bin/activate
python scripts/eval_policy.py \
    --config "$CONFIG" \
    --n-episodes 5 \
    --video-dir "logs/eval_videos/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
CLIENT_STATUS=$?

echo "[client] exited with status $CLIENT_STATUS"
exit $CLIENT_STATUS
