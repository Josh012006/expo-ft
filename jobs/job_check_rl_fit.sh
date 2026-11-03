#!/bin/bash
# Usage:
#   sbatch jobs/job_check_rl_fit.sh [--config PATH] [--checkpoint DIR] [--episode-steps N] [--n-updates N] [--venv NAME]
#     --config          task YAML of the RL run (default: the tactile FORGE one). The server AND the client read it.
#     --checkpoint      SFT checkpoint STEP directory (default: base pi0.5 weights; same memory)
#     --episode-steps   env steps of each of the two real episodes (default: max_steps_per_episode of the YAML)
#     --n-updates       timed steady-state agent.update calls (default 2)
#     --venv            virtualenv of the client (default .venv)
#   (sbatch's own options go BEFORE the script name, e.g. sbatch --output=$SCRATCH/check_%j.out jobs/job_check_rl_fit.sh)
#
# "Will the RL run fit on ONE L40S?", answered in ~15 minutes with nothing trained and nothing saved:
#   1. starts the Isaac Lab server on the card, as the eval job does (YAML robot, tactile sensors included);
#   2. in the SAME card, loads pi0.5 + residual actor + critics exactly as train_pi_robo.py does;
#   3. plays one real episode with the agent's chunks, runs REAL agent.update calls on demonstration data (same shapes
#      as training: 480 transitions, 20 critic steps + 1 actor step), then plays a second episode with the new weights;
#   4. prints the peak of the whole card (every second, all processes) against the card size, and a verdict.
# JAX runs with the exact allocator (XLA_PYTHON_CLIENT_ALLOCATOR=platform), the setting the RL job should use: the
# numbers are what is really needed, with no pool to cap. It was as fast as the default pool in the profile runs.
#
# Output:
#   logs/slurm/check_rl_fit_<jobid>.out        the stage table + the verdict at the end
#   logs/slurm/check_rl_fit_<jobid>_server.log Isaac Lab server log
#   logs/slurm/check_rl_fit_<jobid>_card.csv   memory of the whole card, every second
#   logs/slurm/check_rl_fit_<jobid>_gpumem.csv memory of each process, every 5 s (server vs JAX client)
#
#SBATCH --job-name=check_rl_fit
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

set -u

usage() {
    cat <<'EOF'
Usage: sbatch jobs/job_check_rl_fit.sh [--config PATH] [--checkpoint DIR] [--episode-steps N] [--n-updates N] [--venv NAME]
  --config          task YAML of the RL run (default configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml)
  --checkpoint      SFT checkpoint STEP directory (default: base pi0.5 weights)
  --episode-steps   env steps of each of the two real episodes (default: max_steps_per_episode of the YAML)
  --n-updates       timed steady-state agent.update calls (default 2)
  --venv            virtualenv of the client (default .venv)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

CONFIG="configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml"
CHECKPOINT=""
EPISODE_STEPS=0
N_UPDATES=2
VENV=.venv
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --checkpoint) need_value "$1" "$#" "${2-}"; CHECKPOINT="$2"; shift 2 ;;
        --episode-steps) need_value "$1" "$#" "${2-}"; EPISODE_STEPS="$2"; shift 2 ;;
        --n-updates) need_value "$1" "$#" "${2-}"; N_UPDATES="$2"; shift 2 ;;
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

REPO_ROOT="$HOME/projects/expo-ft"
SERVER_PORT=8102
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to $REPO_ROOT"; exit 1; }
mkdir -p logs/slurm || { echo "FATAL: cannot create logs/slurm"; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG"; exit 1; }

TAG="logs/slurm/${SLURM_JOB_NAME:-check_rl_fit}_${SLURM_JOB_ID:-local}"
SERVER_LOG="${TAG}_server.log"
CARD_LOG="${TAG}_card.csv"
PROC_LOG="${TAG}_gpumem.csv"

# Memory of the whole card every second (what must stay under the card size) and of each process every 5 s.
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$CARD_LOG" 2>&1 &
CARD_PID=$!
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv -l 5 > "$PROC_LOG" 2>&1 &
PROC_PID=$!

SERVER_PID=""
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping server (pid $SERVER_PID, process group)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null          # setsid below: the group holds Isaac Sim / Kit subprocesses too
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
cleanup() {
    kill "$CARD_PID" "$PROC_PID" 2>/dev/null
    stop_server
}
trap cleanup EXIT TERM INT

echo "[server] launching with $CONFIG ..."
(
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
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

# ---- client: the same programs as train_pi_robo.py, on demonstration data, with the server alive on the same card
source scripts/setup_env.sh --venv "$VENV" || exit 1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform      # exact allocator: nvidia-smi = what is really used
CLIENT_ARGS=()
[ -n "$CHECKPOINT" ] && CLIENT_ARGS+=(--checkpoint "$CHECKPOINT")
[ "$EPISODE_STEPS" -gt 0 ] && CLIENT_ARGS+=(--episode-steps "$EPISODE_STEPS")

python3 scripts/profile_memory.py \
    --config "$CONFIG" \
    --with-env \
    --skip-none-branch \
    --n-updates "$N_UPDATES" \
    ${CLIENT_ARGS[@]+"${CLIENT_ARGS[@]}"}
STATUS=$?
echo "[client] profile_memory.py exit status: $STATUS"

# ---- verdict (the server is stopped first so that nothing else is on the card while we read the numbers)
sleep 6                                           # one more card sample after the client exited
stop_server
kill "$CARD_PID" "$PROC_PID" 2>/dev/null
CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo
echo "======================== RL FIT CHECK ========================"
python3 - "$CARD_LOG" "$STATUS" "$CARD_NAME" <<'EOF'
import sys

card_log, status, card_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
used, total = [], None
for line in open(card_log):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 3:
        try:
            used.append(float(parts[1])); total = float(parts[2])
        except ValueError:
            pass
GIB = 1024.0
if not used or total is None:
    print("no card sample was recorded (nvidia-smi unusable): no verdict")
    sys.exit(0)
peak = max(used)
OK_HEADROOM, TIGHT_HEADROOM = 1.5 * GIB, 0.5 * GIB          # MiB
headroom = total - peak
print(f"card                     {card_name}  ({total / GIB:.2f} GiB usable)")
print(f"peak of the whole card   {peak / GIB:.2f} GiB   (JAX + Isaac Lab, sampled every second)")
print(f"headroom                 {headroom / GIB:.2f} GiB   (OK >= {OK_HEADROOM / GIB:.1f}, TIGHT >= {TIGHT_HEADROOM / GIB:.1f})")
if status != 0:
    print(f"verdict                  FAILED: the client exited with status {status}. An out-of-memory error there means it")
    print("                         does NOT fit; anything else is a bug: read the .out / .err of this job.")
elif headroom >= OK_HEADROOM:
    print("verdict                  OK: one card is enough for the server and the RL client (1 environment).")
elif headroom >= TIGHT_HEADROOM:
    print("verdict                  TIGHT: it ran, but with almost no margin. Do not add environments; consider two jobs.")
else:
    print("verdict                  DOES NOT FIT with a safe margin: split the server and the client over two jobs.")
EOF
echo "--- memory the driver attributes to each process over the whole job (max_memory_usage, MiB) ---"
nvidia-smi --query-accounted-apps=pid,max_memory_usage,time --format=csv,noheader 2>/dev/null || echo "(accounting not available)"
echo "per-process timeline: $PROC_LOG    whole card: $CARD_LOG    server log: $SERVER_LOG"
echo "=============================================================="
exit $STATUS
