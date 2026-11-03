#!/bin/bash
# Usage (from ~/projects/expo-ft, on a login node -- plain bash, NOT sbatch):
#   bash jobs/launch_forge_rl.sh [--config PATH] [--sft-checkpoint DIR] [--venv NAME] [--hours N]
#                                [--resume-dir DIR] [--set KEY=VALUE ...] [--cleanup-partition NAME]
#     --config           task YAML (default: the tactile FORGE one), used by the server and the client
#     --sft-checkpoint   SFT checkpoint STEP directory the RL run starts from
#     --venv             virtualenv of the client (default .venv)
#     --hours            time limit of the client; the server gets one hour more (default 40)
#     --resume-dir       RESUME a run after a preemption / crash: its folder, e.g. logs/forge_peg_insert/<run>_rl
#                        (a new server is started; the client restarts from the last agent checkpoint + saved buffer)
#     --set KEY=VALUE    override a top-level field of the YAML (repeatable), for the client only
#     --cleanup-partition  partition of the two tiny cleanup jobs (default main-cpu; "none" = do not pass one)
#
# FORGE RL on two nodes: the simulation on an L40S (jobs/job_forge_server.sh), pi0.5 + RL on an A100L
# (jobs/job_forge_rl_client.sh). The client job only becomes eligible once the server job has STARTED
# (--dependency=after), so the A100L never waits for a server, and it cancels the server job when it ends.
# Two tiny CPU jobs make the coupling hold even when a job is killed without being able to clean up (SIGKILL, node
# failure, time limit): when the client job ends, one cancels the server; when the server job ends, the other cancels
# the client.
set -u
CONFIG="configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml"; SFT=""; VENV=".venv"; HOURS=40
RESUME=""; SETS=(); CLEANUP_PART="main-cpu"
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 && "$2" != --* ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
    case "$1" in
        --config) CONFIG="$2" ;;
        --sft-checkpoint) SFT="$2" ;;
        --venv) VENV="$2" ;;
        --hours) HOURS="$2" ;;
        --resume-dir) RESUME="$2" ;;
        --set) SETS+=(--set "$2") ;;
        --cleanup-partition) CLEANUP_PART="$2" ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
    shift 2
done
cd ~/projects/expo-ft || exit 1
mkdir -p logs/slurm

SERVER_JOB=$(sbatch --parsable --time="$((HOURS + 1)):00:00" jobs/job_forge_server.sh --config "$CONFIG") || { echo "server submission failed"; exit 1; }
SERVER_JOB=${SERVER_JOB%%;*}
CLIENT_JOB=$(sbatch --parsable --time="${HOURS}:00:00" --dependency="after:${SERVER_JOB}" \
    jobs/job_forge_rl_client.sh --server-job "$SERVER_JOB" --venv "$VENV" --config "$CONFIG" \
    ${SFT:+--sft-checkpoint "$SFT"} ${RESUME:+--resume-dir "$RESUME"} ${SETS[@]+"${SETS[@]}"}) || { echo "client submission failed: cancelling the server"; scancel "$SERVER_JOB"; exit 1; }
# Backstops: each tiny job runs when the job it depends on ends, WHATEVER the reason, and cancels the other one.
CLEAN_OPTS=(--parsable --job-name=forge_cleanup --time=00:05:00 --mem=500M --cpus-per-task=1 --output=logs/slurm/%x_%j.out)
[ "$CLEANUP_PART" != "none" ] && CLEAN_OPTS+=(--partition="$CLEANUP_PART")
CLEAN1=$(sbatch "${CLEAN_OPTS[@]}" --dependency="afterany:${CLIENT_JOB}" --wrap="scancel ${SERVER_JOB}") || CLEAN1=""
CLEAN2=$(sbatch "${CLEAN_OPTS[@]}" --dependency="afterany:${SERVER_JOB}" --wrap="scancel ${CLIENT_JOB}") || CLEAN2=""
if [ -z "$CLEAN1" ] || [ -z "$CLEAN2" ]; then
    echo "WARNING: a cleanup job could not be submitted (partition? try --cleanup-partition NAME or none)."
    echo "         The jobs still cancel each other on a normal end / scancel, but not if one is killed hard."
else
    echo "cleanup jobs ${CLEAN1%%;*} (client ends -> cancels server) and ${CLEAN2%%;*} (server ends -> cancels client)"
fi
echo "server job $SERVER_JOB (L40S), client job $CLIENT_JOB (A100L, starts after the server)"
echo "logs: logs/slurm/forge_server_${SERVER_JOB}.out / _server.log   logs/slurm/forge_rl_${CLIENT_JOB}.out"
echo "to stop everything: scancel $CLIENT_JOB  (it cancels the server)   or  scancel $CLIENT_JOB $SERVER_JOB"
