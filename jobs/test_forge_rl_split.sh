#!/bin/bash
# Usage (from ~/projects/expo-ft, on a login node -- plain bash, NOT sbatch; it stays in the foreground and
# submits / watches / cancels jobs, so run it in tmux):
#   bash jobs/test_forge_rl_split.sh [--scenario client-kill|server-kill|all] [--sft-checkpoint DIR]
#                                    [--config PATH] [--venv NAME] [--wait-min N]
#     --scenario        client-kill (default): kill the client mid-run, check the server follows, RESUME, check the
#                       resumed run goes to the end and releases the server.
#                       server-kill: kill the server mid-run, check the client follows.
#     --sft-checkpoint  same as jobs/launch_forge_rl.sh (default: base weights, enough for a plumbing test)
#     --wait-min        longest wait for any single step, in minutes (default 60: includes queue time)
#
# TRIAL RUN of the two-node FORGE RL, short and cheap: 8 episodes of 150 steps, a 2-episode step-0 pass, an agent
# checkpoint every 150 steps, wandb project "expo-ft-split-test", everything under logs/forge_split_test/. It checks:
#   - the server publishes its address and the client connects (server log: "Client connected");
#   - the client runs the step-0 pass and saves an agent checkpoint + replay-buffer files (what a resume needs);
#   - killing one job cancels the other (scancel = what a preemption or a time limit does);
#   - a resume from the folder of the killed run: "Resuming from step ...", the buffer is read back, the run reaches
#     its end ("Saved final agent checkpoint") and the server is released.
# No agent.update happens (it needs 10 episodes; jobs/job_check_rl_fit.sh covers the update on the L40S, and a resumed
# run compiles it in its warm-up, which this test exercises). Prints PASS / FAIL per check, exit status 1 if any failed.
set -u

SCENARIO=client-kill; SFT=""; CONFIG="configs/task/isaaclab/peg_insert_forge_expo_ft_tactile.yaml"; VENV=".venv"; WAIT_MIN=60
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 && "$2" != --* ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
    case "$1" in
        --scenario) SCENARIO="$2" ;;
        --sft-checkpoint) SFT="$2" ;;
        --config) CONFIG="$2" ;;
        --venv) VENV="$2" ;;
        --wait-min) WAIT_MIN="$2" ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
    shift 2
done
case "$SCENARIO" in client-kill|server-kill|all) ;; *) echo "ERROR: unknown scenario $SCENARIO" >&2; exit 1 ;; esac
cd ~/projects/expo-ft || exit 1
: "${SCRATCH:?SCRATCH is not set}"

OUT_DIR=logs/forge_split_test
SMOKE=(--hours 1
       --set max_episodes=8 --set success_rate_window=2 --set checkpoint_interval=150 --set rl_eval_interval=0
       --set "output_dir=\"$OUT_DIR\"" --set 'run_name="split_test"' --set 'project_name="expo-ft-split-test"')
FAILS=0
SUBMITTED=()
# Whatever happens (a FAIL, Ctrl-C), no trial job is left running on the cluster.
trap 'for j in ${SUBMITTED[@]+"${SUBMITTED[@]}"}; do scancel "$j" 2>/dev/null; done' EXIT
trap 'exit 130' INT TERM
ok()   { echo "  PASS  $*"; }
bad()  { echo "  FAIL  $*"; FAILS=$((FAILS + 1)); }
check() { local desc="$1"; shift; if "$@"; then ok "$desc"; else bad "$desc"; fi; }
log()  { echo "[$(date +%H:%M:%S)] $*"; }

job_state() { squeue -h -j "$1" -o %T 2>/dev/null | head -1; }           # empty = the job is gone
job_gone()  { local s; s=$(squeue -h -j "$1" -o %T 2>&1) && [ -z "$s" ]; }  # squeue answered AND no such job
# wait_until <timeout seconds> <command...>: poll every 20 s
wait_until() {
    local timeout="$1"; shift; local t0=$SECONDS
    while ! "$@"; do [ $((SECONDS - t0)) -ge "$timeout" ] && return 1; sleep "$POLL"; done
    return 0
}
LIMIT=$((WAIT_MIN * 60))
POLL=${SPLIT_TEST_POLL:-20}; COUPLE_WAIT=${SPLIT_TEST_COUPLE_WAIT:-600}; SETTLE=${SPLIT_TEST_SETTLE:-30}   # env knobs, for testing the script itself

has_checkpoint() { [ -n "$RUN" ] && find "$RUN/checkpoints" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | grep -q .; }
find_run()      { RUN=$(ls -d "$OUT_DIR"/*-"$1"_rl 2>/dev/null | head -1); [ -n "$RUN" ]; }
is_running()    { [ "$(job_state "$1")" = "RUNNING" ]; }
host_file()     { [ -s "$SCRATCH/forge_server_$1.host" ]; }
in_file()       { grep -q "$1" "$2" 2>/dev/null; }

launch() {   # launch <extra launcher args...> -> sets SERVER, CLIENT
    local out; out=$(bash jobs/launch_forge_rl.sh --config "$CONFIG" --venv "$VENV" ${SFT:+--sft-checkpoint "$SFT"} "${SMOKE[@]}" "$@") \
        || { echo "$out"; echo "launcher failed"; exit 1; }
    echo "$out"
    SERVER=$(echo "$out" | sed -n 's/^server job \([0-9]*\) .*/\1/p'); CLIENT=$(echo "$out" | sed -n 's/.*client job \([0-9]*\) .*/\1/p')
    [ -n "$SERVER" ] && [ -n "$CLIENT" ] || { echo "could not read the job ids"; exit 1; }
    SUBMITTED+=("$SERVER" "$CLIENT")
}

# bring a launched pair up to "the client has saved an agent checkpoint and the buffer files exist"
reach_checkpoint() {
    log "waiting for the server to publish its address (queue + boot)..."
    wait_until "$LIMIT" host_file "$SERVER" && ok "server $SERVER published its address: $(cat "$SCRATCH/forge_server_$SERVER.host")" \
        || { bad "server $SERVER never published its address (see logs/slurm/forge_server_${SERVER}_server.log)"; return 1; }
    log "waiting for the client to start and create its run folder..."
    wait_until "$LIMIT" find_run "$CLIENT" && ok "client $CLIENT created $RUN" \
        || { bad "client $CLIENT never created its run folder (see logs/slurm/forge_rl_${CLIENT}.out)"; return 1; }
    check "client connected to the server" wait_until "$COUPLE_WAIT" in_file "Client connected" "logs/slurm/forge_server_${SERVER}_server.log"
    log "waiting for the first agent checkpoint (step-0 pass + 1 training episode, a few minutes after the client starts)..."
    wait_until "$LIMIT" has_checkpoint && ok "agent checkpoint written: $(ls "$RUN/checkpoints" | tr '\n' ' ')" \
        || { bad "no agent checkpoint (see logs/slurm/forge_rl_${CLIENT}.out)"; return 1; }
    sleep "$SETTLE"   # let the buffer files of the steps after the checkpoint accumulate too
    check "replay-buffer files saved ($(ls "$RUN/checkpoints/buffers" 2>/dev/null | wc -l) files)" \
        test "$(ls "$RUN/checkpoints/buffers" 2>/dev/null | wc -l)" -gt 150
    check "wandb id saved for the resume" test -s "$RUN/checkpoints/wandb_id.txt"
}

scenario_client_kill() {
    echo; echo "=== scenario client-kill: kill the client mid-run, then resume ==="
    launch
    reach_checkpoint || return
    log "scancel client $CLIENT (what a preemption does)"
    scancel "$CLIENT"
    wait_until "$COUPLE_WAIT" job_gone "$CLIENT"
    log "waiting up to 10 min for the server $SERVER to follow (client trap, else the cleanup job)..."
    check "server $SERVER released after the client was killed" wait_until "$COUPLE_WAIT" job_gone "$SERVER"
    local killed_run="$RUN"

    echo "--- resume from $killed_run"
    launch --resume-dir "$killed_run"
    local server2=$SERVER client2=$CLIENT
    wait_until "$LIMIT" host_file "$server2" || { bad "resume: server $server2 never published its address"; return; }
    check "resume: client $client2 says it resumes" wait_until "$LIMIT" in_file "Resuming from step" "logs/slurm/forge_rl_${client2}.out"
    log "waiting for the resumed run to finish (restore + warm-up compile + the remaining episodes)..."
    check "resume: run reached its end" wait_until "$LIMIT" in_file "Saved final agent checkpoint" "logs/slurm/forge_rl_${client2}.out"
    log "waiting up to 10 min for the client to end and the server to be released..."
    check "resume: client $client2 ended" wait_until "$COUPLE_WAIT" job_gone "$client2"
    check "resume: server $server2 released after a normal end" wait_until "$COUPLE_WAIT" job_gone "$server2"
    check "resume: same run folder reused (no new folder)" test "$(ls -d "$OUT_DIR"/*_rl | wc -l)" -eq 1
    echo "  logs: logs/slurm/forge_rl_${client2}.out  (grep -n 'Resuming\|warmup\|Saved' on it)"
}

scenario_server_kill() {
    echo; echo "=== scenario server-kill: kill the server mid-run ==="
    launch
    reach_checkpoint || return
    log "scancel server $SERVER"
    scancel "$SERVER"
    wait_until "$COUPLE_WAIT" job_gone "$SERVER"
    log "waiting up to 10 min for the client $CLIENT to follow (cleanup job)..."
    check "client $CLIENT cancelled after the server was killed" wait_until "$COUPLE_WAIT" job_gone "$CLIENT"
}

case "$SCENARIO" in
    client-kill) scenario_client_kill ;;
    server-kill) scenario_server_kill ;;
    all) scenario_client_kill; rm -rf "$OUT_DIR"; scenario_server_kill ;;
esac
echo
[ "$FAILS" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "$FAILS CHECK(S) FAILED -- read the logs named above"
echo "(trial runs are under $OUT_DIR/ and in the wandb project expo-ft-split-test: delete them when done)"
exit $((FAILS > 0))
