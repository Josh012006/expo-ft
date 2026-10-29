#!/bin/bash
# Fixed-seed evaluation CURVE on FORGE PegInsert: the base model (zero-shot) and EVERY numeric checkpoint
# sub-folder (200, 400, ...) of an SFT run, all on the SAME episode seeds. Same job as run_forge_eval_job.sh
# (Isaac Lab server + client on one L40S) but the client is scripts/eval_curve.py instead of eval_policy.py.
#
# Usage:
#   sbatch jobs/run_forge_eval_curve_job.sh --checkpoints-dir DIR [--n-episodes N] [--steps "S1 S2"] [--seed N]
#        [--robot NAME] [--max-steps N] [--server-port P] [--save-videos] [--video-dir NAME|PATH] [--skip-base] [--output-dir DIR]
#     --checkpoints-dir  (required) folder that contains the numeric step folders, e.g.
#                        logs/forge_peg_insert/<run>/sft/expo_pi05_droid_lora_finetune_sft_joint_state_delta/<exp_name>
#     --n-episodes       episodes per checkpoint, same seeds for all (default 200)
#     --save-videos      switch: also record the rollout videos of every checkpoint (off: only the curve is produced)
#     --video-dir        where the videos go (implies --save-videos): <video-dir>/<step>/ for each checkpoint, with its
#                        peg_socket_metrics.jsonl. A bare name goes under logs/eval_videos/ (--video-dir my_run ->
#                        logs/eval_videos/my_run/200/ ...); a path is used as is. Default: one timestamped folder per
#                        checkpoint in logs/eval_videos/
#     --skip-base        switch: do not evaluate the zero-shot base model
#     --steps            only these checkpoints, e.g. --steps 1000 or --steps "1000 1600" (quoted); implies --skip-base
#     --seed             master seed of the fixed episode list (default 42). The seeds of a run are written to
#                        <output-dir>/episode_seeds.json: the same --seed and --n-episodes give the same seeds
#     --robot            robot of the simulation: franka | robotiq_2f85 | robotiq_2f85_tsf (default: the YAML's, franka).
#                        MUST be the robot the checkpoints were trained on (the Robotiq SFT runs: robotiq_2f85_tsf).
#                        Server AND client use it, through the copy of the YAML in <output-dir>/config/
#     --max-steps        episode length in steps (default: the YAML's max_steps_per_episode, 150). Server AND client
#                        use it: the script writes a copy of the YAML with this value into <output-dir>/config/
#     --server-port      port of the Isaac Lab server (default: the YAML's server_port, 8102). Needed when two of
#                        these jobs can land on the SAME node (the server listens on localhost of the node)
#     --output-dir       where results/, logs/, episode_seeds.json, curve.json and curve.png/.pdf go (default: --checkpoints-dir).
#                        Give a folder of its own whenever --seed, --max-steps or --steps differ from an earlier run,
#                        otherwise its episode_seeds.json / results/ would be reused or mixed up
#   (sbatch's own options, e.g. --time, go BEFORE the script name; everything after it is read by this script)
#
# Duration: an episode takes ~35 s on the L40S, so 200 episodes ~ 2 h per checkpoint (+ ~5 min start-up).
# 20 checkpoints + base ~ 40 h. The sweep is resumable: a checkpoint that already has results/<step>.json is
# skipped, so if the job hits its time limit just resubmit the same command and it continues.
# curve.png / curve.pdf are rebuilt after every checkpoint, a killed sweep still leaves an up-to-date plot.
#
#SBATCH --job-name=forge-eval-curve
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --no-requeue

# Server + client, same node, same GPU, one allocation. The server stays up for the whole sweep; eval_curve.py
# starts one eval_policy.py process per checkpoint (fresh GPU memory each time) and they all connect to it.
#
# Cancel-safety: `trap cleanup EXIT TERM INT` below fires on normal script
# completion AND on scancel (SLURM sends TERM, with a grace period before
# KILL) — so the server and its Isaac Sim subprocesses always get torn down,
# never left pending/orphaned on the node.

set -u  # catch unset-variable typos; NOT set -e (would fight the trap/background logic)

usage() {
    cat <<'EOF'
Usage: sbatch jobs/run_forge_eval_curve_job.sh --checkpoints-dir DIR [--n-episodes N] [--steps "S1 S2"] [--seed N]
         [--robot NAME] [--max-steps N] [--server-port P] [--save-videos] [--video-dir NAME|PATH] [--skip-base] [--output-dir DIR]
  --checkpoints-dir  (required) folder with the numeric step sub-folders of an SFT run
  --n-episodes       episodes per checkpoint, same seeds for all (default 200)
  --save-videos      switch: also record the rollout videos of every checkpoint
  --video-dir        where the videos go (implies --save-videos): <video-dir>/<step>/ per checkpoint; a bare
                     name goes under logs/eval_videos/ (default: one timestamped folder per checkpoint)
  --skip-base        switch: do not evaluate the zero-shot base model
  --steps            only these checkpoints, e.g. --steps 1000 or --steps "1000 1600" (implies --skip-base)
  --seed             master seed of the fixed episode list (default 42)
  --robot            franka | robotiq_2f85 | robotiq_2f85_tsf (default: the YAML's, franka); the robot of the training
  --max-steps        episode length in steps (default: the YAML's, 150)
  --server-port      port of the Isaac Lab server (default: the YAML's, 8102)
  --output-dir       where results/, logs/, episode_seeds.json, curve.* go (default: --checkpoints-dir;
                     give a folder of its own when --seed / --max-steps / --steps differ from an earlier run)
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

CHECKPOINTS_DIR=""
N_EPISODES=200
OUTPUT_DIR=""
SAVE_VIDEOS=0
VIDEO_DIR=""
SKIP_BASE=0
STEPS=""
MASTER_SEED=""
MAX_STEPS=""
PORT_OVERRIDE=""
ROBOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints-dir) need_value "$1" "$#" "${2-}"; CHECKPOINTS_DIR="$2"; shift 2 ;;
        --n-episodes) need_value "$1" "$#" "${2-}"; N_EPISODES="$2"; shift 2 ;;
        --output-dir) need_value "$1" "$#" "${2-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --save-videos) SAVE_VIDEOS=1; shift ;;
        --video-dir) need_value "$1" "$#" "${2-}"; VIDEO_DIR="$2"; shift 2 ;;
        --skip-base) SKIP_BASE=1; shift ;;
        --steps) need_value "$1" "$#" "${2-}"; STEPS="$2"; shift 2 ;;
        --seed) need_value "$1" "$#" "${2-}"; MASTER_SEED="$2"; shift 2 ;;
        --robot) need_value "$1" "$#" "${2-}"; ROBOT="$2"; shift 2 ;;
        --max-steps) need_value "$1" "$#" "${2-}"; MAX_STEPS="$2"; shift 2 ;;
        --server-port) need_value "$1" "$#" "${2-}"; PORT_OVERRIDE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
if [[ -z "$CHECKPOINTS_DIR" ]]; then echo "ERROR: --checkpoints-dir is required" >&2; usage >&2; exit 1; fi
if [[ -n "$VIDEO_DIR" ]]; then SAVE_VIDEOS=1; fi    # a video folder means videos

REPO_ROOT="$HOME/projects/expo-ft"
CONFIG="configs/task/isaaclab/peg_insert_forge_pi05.yaml"
SERVER_LOG="logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_server.log"

cd "$REPO_ROOT" || { echo "FATAL: cannot cd to $REPO_ROOT"; exit 1; }
mkdir -p logs/slurm || { echo "FATAL: cannot create logs/slurm"; exit 1; }

# Checked BEFORE starting the server (no 5 min of Isaac Sim start-up for a typo in the path).
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "FATAL: --checkpoints-dir $CHECKPOINTS_DIR not found"
    exit 1
fi
N_CKPT=$(find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | wc -l)
if [ "$N_CKPT" -eq 0 ]; then
    echo "FATAL: no numeric checkpoint folder (200, 400, ...) in $CHECKPOINTS_DIR"
    exit 1
fi

# --robot / --max-steps / --server-port: a copy of the YAML (same file name) in <output-dir>/config/ is used by the server AND the
# client, so both always agree and the run keeps the exact config it used. Without them the repo's YAML is used as is.
if [ -n "$ROBOT" ] || [ -n "$MAX_STEPS" ] || [ -n "$PORT_OVERRIDE" ]; then
    DERIVED_DIR="${OUTPUT_DIR:-$CHECKPOINTS_DIR}/config"
    mkdir -p "$DERIVED_DIR" || { echo "FATAL: cannot create $DERIVED_DIR"; exit 1; }
    DERIVED="$DERIVED_DIR/$(basename "$CONFIG")"
    cp "$CONFIG" "$DERIVED"
    if [ -n "$ROBOT" ]; then
        case "$ROBOT" in franka|robotiq_2f85|robotiq_2f85_tsf) ;; *) echo "FATAL: --robot must be franka, robotiq_2f85 or robotiq_2f85_tsf (got $ROBOT)"; exit 1 ;; esac
        sed -i -E "s/^(robot:)[[:space:]]*.*/\1 \"$ROBOT\"/" "$DERIVED"
        grep -q -E "^robot: \"$ROBOT\"" "$DERIVED" || { echo "FATAL: could not set robot in $DERIVED"; exit 1; }
    fi
    if [ -n "$MAX_STEPS" ]; then
        sed -i -E "s/^(max_steps_per_episode:)[[:space:]]*[0-9]+/\1 $MAX_STEPS/" "$DERIVED"
        grep -q -E "^max_steps_per_episode: $MAX_STEPS( |$)" "$DERIVED" || { echo "FATAL: could not set max_steps_per_episode in $DERIVED"; exit 1; }
    fi
    if [ -n "$PORT_OVERRIDE" ]; then
        sed -i -E "s/^(server_port:)[[:space:]]*[0-9]+/\1 $PORT_OVERRIDE/" "$DERIVED"
        grep -q -E "^server_port: $PORT_OVERRIDE( |$)" "$DERIVED" || { echo "FATAL: could not set server_port in $DERIVED"; exit 1; }
    fi
    CONFIG="$DERIVED"
    echo "[config] derived config: $CONFIG"
fi
SERVER_PORT=$(grep -E "^server_port:" "$CONFIG" | head -1 | sed -E 's/^server_port:[[:space:]]*([0-9]+).*/\1/')
[ -n "$SERVER_PORT" ] || { echo "FATAL: no server_port in $CONFIG"; exit 1; }
echo "[config] $CONFIG: robot=$(grep -E '^robot:' "$CONFIG" | head -1 | sed -E 's/^robot:[[:space:]]*"?([A-Za-z0-9_]+)"?.*/\1/'), max_steps_per_episode=$(grep -E '^max_steps_per_episode:' "$CONFIG" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/'), server_port=$SERVER_PORT"

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

echo "[client] eval curve: $N_CKPT checkpoints in $CHECKPOINTS_DIR (steps: ${STEPS:-all}), $N_EPISODES episodes each, seed ${MASTER_SEED:-42 (default)}, videos: $SAVE_VIDEOS (dir: ${VIDEO_DIR:-default}), skip base: $SKIP_BASE"
source .venv/bin/activate
CURVE_ARGS=()
if [ "$SAVE_VIDEOS" -eq 1 ]; then CURVE_ARGS+=(--save-videos); fi
if [ -n "$VIDEO_DIR" ]; then CURVE_ARGS+=(--video-dir "$VIDEO_DIR"); fi
if [ "$SKIP_BASE" -eq 1 ]; then CURVE_ARGS+=(--skip-base); fi
if [ -n "$OUTPUT_DIR" ]; then CURVE_ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if [ -n "$MASTER_SEED" ]; then CURVE_ARGS+=(--seed "$MASTER_SEED"); fi
if [ -n "$STEPS" ]; then CURVE_ARGS+=(--steps $STEPS); fi    # unquoted on purpose: "1000 1600" -> two values
python scripts/eval_curve.py \
    --config "$CONFIG" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --n-episodes "$N_EPISODES" \
    ${CURVE_ARGS[@]+"${CURVE_ARGS[@]}"}
CLIENT_STATUS=$?

echo "[client] exited with status $CLIENT_STATUS"
exit $CLIENT_STATUS
