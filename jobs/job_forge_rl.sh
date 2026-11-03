#!/bin/bash
# Usage:
#   sbatch [sbatch options] jobs/job_forge_rl.sh --sft-checkpoint DIR [--demos-dir DIR] [--offline-ratio X] [--tag NAME]
#                                                [--resume-dir DIR] [--set KEY=VALUE ...] [--config PATH] [--venv NAME]
#     --sft-checkpoint  SFT checkpoint STEP directory the RL run starts from (the folder that contains params/). Required.
#     --demos-dir       demos of the YAML override (demos_dir). Must be the demos of that SFT checkpoint.
#                       With offline_ratio 0.0 they are only read for the shapes: they never enter a buffer.
#     --offline-ratio   fraction of each batch drawn from the demos (offline_ratio of the YAML, 0.5)
#     --tag             name of the run: run_name = forge_rl_<tag> (folder, wandb); default: none (YAML's run_name)
#     --resume-dir      RESUME a run (preemption, crash): its folder, e.g. logs/forge_peg_insert/<run>_rl
#     --set KEY=VALUE   override any other top-level field of the YAML (repeatable; the key must exist in it)
#     --config          task YAML (default: RL on the task alone, no tactile, no force)
#     --venv            virtualenv of the client (default .venv)
#   Normally called by jobs/launch_forge_rl_4.sh.
#
# FORGE RL with EVERYTHING on ONE L40S: the Isaac Lab server and the JAX client (pi0.5 + residual actor + critics) share
# the card (measured peak 43.7 of 45 GiB with the tactile robot: JAX runs with the exact allocator,
# XLA_PYTHON_CLIENT_ALLOCATOR=platform, which is what makes it fit). No evaluation during training (rl_eval_interval: 0).
# The YAML is copied to logs/slurm/<jobname>_<jobid>_<yaml name> with the overrides and a free server port (20000 + id % 20000:
# L40S nodes are shared and 8102 may be taken); the YAML in the repo is untouched. Server and client are in the same
# job: when the job ends (finished, scancel, preemption, time limit) both go.
#
# Output:
#   logs/slurm/<jobname>_<jobid>.out            client (training) log
#   logs/slurm/<jobname>_<jobid>_server.log     Isaac Lab server log
#   logs/slurm/<jobname>_<jobid>_card.csv       memory of the whole card, every 30 s
#   logs/slurm/<jobname>_<jobid>_<yaml name>    the YAML actually used
#
#SBATCH --job-name=forge_rl
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=60:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

CONFIG="configs/task/isaaclab/peg_insert_forge_expo_ft.yaml"
VENV=.venv
SFT_CHECKPOINT=""; DEMOS_DIR=""; OFFLINE_RATIO=""; TAG=""; RESUME_DIR=""
SETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sft-checkpoint) need_value "$1" "$#" "${2-}"; SFT_CHECKPOINT="$2"; shift 2 ;;
        --demos-dir) need_value "$1" "$#" "${2-}"; DEMOS_DIR="$2"; shift 2 ;;
        --offline-ratio) need_value "$1" "$#" "${2-}"; OFFLINE_RATIO="$2"; shift 2 ;;
        --tag) need_value "$1" "$#" "${2-}"; TAG="$2"; shift 2 ;;
        --resume-dir) need_value "$1" "$#" "${2-}"; RESUME_DIR="$2"; shift 2 ;;
        --set) need_value "$1" "$#" "${2-}"; SETS+=("$2"); shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        -h|--help) sed -n 2,14p "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done
[ -n "$SFT_CHECKPOINT" ] || { echo "ERROR: --sft-checkpoint is required"; exit 1; }

REPO_ROOT="$HOME/projects/expo-ft"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to $REPO_ROOT"; exit 1; }
mkdir -p logs/slurm || { echo "FATAL: cannot create logs/slurm"; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG"; exit 1; }
[ -d "$SFT_CHECKPOINT/params" ] || { echo "FATAL: $SFT_CHECKPOINT/params not found: give the checkpoint STEP directory"; exit 1; }
[ -z "$DEMOS_DIR" ] || [ -d "$DEMOS_DIR" ] || { echo "FATAL: demos dir not found: $DEMOS_DIR"; exit 1; }
if [ -n "$RESUME_DIR" ]; then
    [ -d "$RESUME_DIR/checkpoints" ] || { echo "FATAL: $RESUME_DIR/checkpoints not found: not a run folder to resume"; exit 1; }
fi

JOB_ID="${SLURM_JOB_ID:-local}"
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
TAGNAME="logs/slurm/${SLURM_JOB_NAME:-forge_rl}_${JOB_ID}"
SERVER_LOG="${TAGNAME}_server.log"
CARD_LOG="${TAGNAME}_card.csv"
RUN_CONFIG="${TAGNAME}_$(basename "$CONFIG")"

# ---- the YAML of this run
ARGS=("server_host=\"localhost\"" "server_port=$PORT")
[ -n "$DEMOS_DIR" ]     && ARGS+=("demos_dir=\"$DEMOS_DIR\"")
[ -n "$OFFLINE_RATIO" ] && ARGS+=("offline_ratio=$OFFLINE_RATIO")
[ -n "$TAG" ]           && ARGS+=("run_name=\"forge_rl_$TAG\"")
[ -n "$RESUME_DIR" ]    && ARGS+=("rl_resume_dir=\"$RESUME_DIR\"")
ARGS+=("${SETS[@]+"${SETS[@]}"}")
python3 - "$CONFIG" "$RUN_CONFIG" "${ARGS[@]}" <<'PYEOF'
import re, sys
src, dst = sys.argv[1:3]
text = open(src).read()
for item in sys.argv[3:]:
    if not item:
        continue
    key, sep, value = item.partition("=")
    assert sep and key, f"expected KEY=VALUE, got {item!r}"
    text, n = re.subn(rf"^{re.escape(key)}:[^\n#]*", lambda m: f"{key}: {value} ", text, flags=re.M)
    assert n == 1, f"{key}: expected exactly one top-level line in {src}, found {n}"
open(dst, "w").write(text)
PYEOF
[ $? -eq 0 ] || { echo "FATAL: could not write $RUN_CONFIG"; exit 1; }
echo "[config] $RUN_CONFIG:"
grep -n "^demos_dir\|^offline_ratio\|^rl_seed_demos_online\|^run_name\|^rl_resume_dir\|^server_port\|^rl_eval_interval\|^use_tactile\|^use_force\|^success_rate_window\|^max_episodes" "$RUN_CONFIG"
echo "[run] SFT checkpoint: $SFT_CHECKPOINT"

nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -l 30 > "$CARD_LOG" 2>&1 &
CARD_PID=$!

SERVER_PID=""
cleanup() {
    kill "$CARD_PID" 2>/dev/null
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping server (pid $SERVER_PID, process group)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null          # setsid below: the group holds Isaac Sim / Kit subprocesses too
        for _ in $(seq 1 15); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL -"$SERVER_PID" 2>/dev/null
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# ---- server (same card)
echo "[server] launching on port $PORT ..."
(
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    source .venv-isaaclab/bin/activate
    setsid python -m expo_ft.env.isaaclab.forge_server --config "$RUN_CONFIG"
) > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
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
echo "[server] ready after $((i * 5))s."

# ---- client
source scripts/setup_env.sh --venv "$VENV" || exit 1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform      # exact allocator: this is what lets JAX and the server share the 45 GiB card
python3 scripts/run_pipeline.py \
    --config "$RUN_CONFIG" \
    --stage rl \
    --sft-checkpoint "$SFT_CHECKPOINT"
STATUS=$?
echo "[client] exit status: $STATUS"
exit $STATUS
