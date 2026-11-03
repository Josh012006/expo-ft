#!/bin/bash
# Usage:
#   sbatch [sbatch options] jobs/job_forge_server.sh [--config PATH]
#     --config   task YAML of the RL run (default: the tactile FORGE one). Same file as the client's.
#   Normally NOT called by hand: jobs/launch_forge_rl.sh submits it together with jobs/job_forge_rl_client.sh.
#
# Isaac Lab / FORGE server alone on an L40S, for a client (JAX + pi0.5, jobs/job_forge_rl_client.sh) running in
# ANOTHER job on an A100L. Differences with the same-node setup (jobs/run_forge_eval_job.sh):
#   - the server listens on every interface (--set server_host=0.0.0.0): the YAML's "localhost" would refuse the client;
#   - the port depends on the job id (20000 + id % 20000): L40S nodes are shared and 8102 may already be taken;
#   - once the server accepts connections, "<hostname> <port>" is written to $SCRATCH/forge_server_<jobid>.host:
#     the client reads its server's address from there.
# Stays up until the client job cancels it (or until --time).
#
# Output:
#   logs/slurm/forge_server_<jobid>.out         this script
#   logs/slurm/forge_server_<jobid>_server.log  Isaac Lab server log
#   logs/slurm/forge_server_<jobid>_gpumem.csv  memory of the card, every 10 s
#
#SBATCH --job-name=forge_server
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=41:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u

CONFIG="configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) [[ $# -ge 2 && "$2" != --* ]] || { echo "ERROR: --config needs a value" >&2; exit 1; }; CONFIG="$2"; shift 2 ;;
        -h|--help) sed -n 2,6p "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$HOME/projects/expo-ft"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to $REPO_ROOT"; exit 1; }
mkdir -p logs/slurm || { echo "FATAL: cannot create logs/slurm"; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG"; exit 1; }
: "${SCRATCH:?SCRATCH is not set}"

JOB_ID="${SLURM_JOB_ID:-local}"
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
HOST_FILE="$SCRATCH/forge_server_${JOB_ID}.host"
SERVER_LOG="logs/slurm/${SLURM_JOB_NAME:-forge_server}_${JOB_ID}_server.log"
GPU_LOG="logs/slurm/${SLURM_JOB_NAME:-forge_server}_${JOB_ID}_gpumem.csv"
rm -f "$HOST_FILE"

nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -l 10 > "$GPU_LOG" 2>&1 &
GPU_PID=$!

SERVER_PID=""
cleanup() {
    rm -f "$HOST_FILE"
    kill "$GPU_PID" 2>/dev/null
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping server (pid $SERVER_PID, process group)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null          # setsid below: the group holds Isaac Sim / Kit subprocesses too
        for _ in $(seq 1 15); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL -"$SERVER_PID" 2>/dev/null
    fi
}
trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT

echo "[server] node $(hostname), port $PORT, config $CONFIG"
(
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    source .venv-isaaclab/bin/activate
    setsid python -m expo_ft.env.isaaclab.forge_server --config "$CONFIG" \
        --set server_host=0.0.0.0 --set server_port="$PORT"
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[server] pid=$SERVER_PID, log=$SERVER_LOG"

READY=0
for i in $(seq 1 60); do  # 60 x 5s = 5 min ceiling
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[server] process died before becoming ready -- see $SERVER_LOG"; exit 1
    fi
    if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(1)
sys.exit(0 if s.connect_ex(('localhost', $PORT)) == 0 else 1)" 2>/dev/null; then
        READY=1; break
    fi
    sleep 5
done
[ "$READY" -eq 1 ] || { echo "[server] never became ready within 5 min -- see $SERVER_LOG"; exit 1; }

echo "$(hostname) $PORT" > "$HOST_FILE.tmp" && mv "$HOST_FILE.tmp" "$HOST_FILE"
echo "[server] ready after $((i * 5))s; address written to $HOST_FILE: $(cat "$HOST_FILE")"

wait "$SERVER_PID"
STATUS=$?
echo "[server] exited with status $STATUS"
exit $STATUS
