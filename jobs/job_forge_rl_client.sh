#!/bin/bash
# Usage:
#   sbatch [sbatch options] jobs/job_forge_rl_client.sh --server-job ID [--venv NAME] [--config PATH] [--sft-checkpoint DIR]
#                                                       [--resume-dir DIR] [--set KEY=VALUE ...]
#     --server-job      job id of the running jobs/job_forge_server.sh (required)
#     --venv            virtualenv to activate (default .venv)
#     --config          task YAML (default: the tactile FORGE one). Must be the one the server was started with.
#     --sft-checkpoint  SFT checkpoint STEP directory the RL run starts from (default: the config's base weights)
#     --resume-dir      run folder to resume (sets rl_resume_dir in the YAML copy), e.g. logs/forge_peg_insert/<run>_rl
#     --set KEY=VALUE   override a top-level field of the YAML copy (repeatable; the key must exist in the YAML)
#   Normally NOT called by hand: jobs/launch_forge_rl.sh submits it together with the server.
#
# RL client (pi0.5 + residual actor + critics, JAX) alone on an A100L; the simulation runs in the server job on an L40S.
# This script reads the server's address (written by the server job in $SCRATCH/forge_server_<id>.host), writes a copy
# of the YAML with server_host / server_port set to it (plus --resume-dir / --set, if any; the copy is
# logs/slurm/<jobname>_<jobid>_<yaml name>; the YAML in the repo is untouched), and runs the same command as
# jobs/job_rl.sh on that copy. When it ends it cancels the server job so that the L40S is released (if it is killed
# without being able to, the cleanup job of jobs/launch_forge_rl.sh does it).
#
#SBATCH --job-name=forge_rl
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=a100l:1
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

VENV=.venv
CONFIG=configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml
SFT_CHECKPOINT=
SERVER_JOB=
RESUME_DIR=
SETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-job) need_value "$1" "$#" "${2-}"; SERVER_JOB="$2"; shift 2 ;;
        --venv) need_value "$1" "$#" "${2-}"; VENV="$2"; shift 2 ;;
        --config) need_value "$1" "$#" "${2-}"; CONFIG="$2"; shift 2 ;;
        --sft-checkpoint) need_value "$1" "$#" "${2-}"; SFT_CHECKPOINT="$2"; shift 2 ;;
        --resume-dir) need_value "$1" "$#" "${2-}"; RESUME_DIR="$2"; shift 2 ;;
        --set) need_value "$1" "$#" "${2-}"; SETS+=("$2"); shift 2 ;;
        -h|--help) sed -n 2,12p "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done
[ -n "$SERVER_JOB" ] || { echo "ERROR: --server-job is required"; exit 1; }

cd ~/projects/expo-ft || { echo "FATAL: cannot cd to ~/projects/expo-ft"; exit 1; }
mkdir -p logs/slurm
[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG"; exit 1; }
if [ -n "$RESUME_DIR" ]; then
    [ -d "$RESUME_DIR/checkpoints" ] || { echo "FATAL: $RESUME_DIR/checkpoints not found: not a run folder to resume"; exit 1; }
    SETS+=("rl_resume_dir=\"$RESUME_DIR\"")
fi
: "${SCRATCH:?SCRATCH is not set}"

# Whatever happens below, free the L40S.
release_server() {
    echo "[client] cancelling server job $SERVER_JOB"
    scancel "$SERVER_JOB" 2>/dev/null
}
trap release_server EXIT
trap 'exit 143' TERM INT

# ---- address of the server
HOST_FILE="$SCRATCH/forge_server_${SERVER_JOB}.host"
echo "[client] waiting for $HOST_FILE ..."
for _ in $(seq 1 360); do   # 360 x 5 s = 30 min: the server job has already started (dependency), it only has to boot
    [ -s "$HOST_FILE" ] && break
    if STATE=$(squeue -h -j "$SERVER_JOB" -o %T 2>/dev/null) && [ -z "$STATE" ]; then
        echo "FATAL: server job $SERVER_JOB is no longer in the queue (see logs/slurm/forge_server_${SERVER_JOB}.out)"; exit 1
    fi
    sleep 5
done
[ -s "$HOST_FILE" ] || { echo "FATAL: server $SERVER_JOB never published its address (see logs/slurm/forge_server_${SERVER_JOB}_server.log)"; exit 1; }
read -r SERVER_HOST SERVER_PORT < "$HOST_FILE"
echo "[client] server: $SERVER_HOST:$SERVER_PORT"
python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(5)
r = s.connect_ex(('$SERVER_HOST', $SERVER_PORT)); print('[client] connection test:', 'ok' if r == 0 else f'FAILED (errno {r})'); sys.exit(0 if r == 0 else 1)
" || { echo "FATAL: cannot reach $SERVER_HOST:$SERVER_PORT from this node (firewall between nodes?)"; exit 1; }

# ---- copy of the YAML pointing at that server
RUN_CONFIG="logs/slurm/${SLURM_JOB_NAME:-forge_rl}_${SLURM_JOB_ID:-local}_$(basename "$CONFIG")"
python3 - "$CONFIG" "$RUN_CONFIG" "server_host=\"$SERVER_HOST\"" "server_port=$SERVER_PORT" ${SETS[@]+"${SETS[@]}"} <<'PYEOF'
import re, sys
src, dst = sys.argv[1:3]
text = open(src).read()
for item in sys.argv[3:]:
    key, sep, value = item.partition("=")
    assert sep and key, f"expected KEY=VALUE, got {item!r}"
    text, n = re.subn(rf"^{re.escape(key)}:[^\n#]*", lambda m: f"{key}: {value} ", text, flags=re.M)
    assert n == 1, f"{key}: expected exactly one top-level line in {src}, found {n}"
open(dst, "w").write(text)
PYEOF
[ $? -eq 0 ] || { echo "FATAL: could not write $RUN_CONFIG"; exit 1; }
grep -n "^server_host\|^server_port\|^rl_resume_dir" "$RUN_CONFIG"

source scripts/setup_env.sh --venv "$VENV" || exit 1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97     # the card is for JAX alone here, as in jobs/job_rl.sh
python3 scripts/run_pipeline.py \
    --config "$RUN_CONFIG" \
    --stage rl \
    ${SFT_CHECKPOINT:+--sft-checkpoint "$SFT_CHECKPOINT"}
