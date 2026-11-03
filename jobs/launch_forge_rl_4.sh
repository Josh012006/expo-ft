#!/bin/bash
# Usage (from ~/projects/expo-ft, on a login node -- plain bash, NOT sbatch):
#   bash jobs/launch_forge_rl_4.sh --clean-ckpt DIR --noisy-ckpt DIR --teleop-ckpt DIR [--nodemo-from clean|noisy|teleop]
#                                  [--only LIST] [--hours N] [--set KEY=VALUE ...]
#     --clean-ckpt / --noisy-ckpt / --teleop-ckpt   SFT checkpoint STEP directories (the folder that contains params/):
#                                  clean = scripted demos, best checkpoint; noisy = noisy scripted demos, best checkpoint;
#                                  teleop = teleoperated demos, the checkpoint just at >= 40 %.
#     --nodemo-from     SFT the no-demos run starts from (default noisy); its demos are only read for the shapes
#     --only            comma-separated subset of: nodemo,clean,noisy,teleop (default: all four)
#     --hours           time limit of each job (default 60)
#     --set KEY=VALUE   override a YAML field in the four runs (repeatable)
#
# The four FORGE RL runs, RL on the task alone (no tactile, no force), each on its own L40S (jobs/job_forge_rl.sh):
#   nodemo   offline_ratio 0.0, no demo in any buffer
#   clean    offline_ratio 0.5, demos/isaaclab/ForgePegInsert_Robotiq_scripted
#   noisy    offline_ratio 0.5, demos/isaaclab/ForgePegInsert_Robotiq_noisy
#   teleop   offline_ratio 0.5, demos/isaaclab/ForgePegInsert_Robotiq_teleop
set -u
CLEAN=""; NOISY=""; TELEOP=""; FROM="noisy"; ONLY="nodemo,clean,noisy,teleop"; HOURS=60; SETS=()
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 && "$2" != --* ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }
    case "$1" in
        --clean-ckpt) CLEAN="$2" ;;
        --noisy-ckpt) NOISY="$2" ;;
        --teleop-ckpt) TELEOP="$2" ;;
        --nodemo-from) FROM="$2" ;;
        --only) ONLY="$2" ;;
        --hours) HOURS="$2" ;;
        --set) SETS+=(--set "$2") ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
    shift 2
done
case "$FROM" in clean|noisy|teleop) ;; *) echo "ERROR: --nodemo-from must be clean, noisy or teleop" >&2; exit 1 ;; esac
cd ~/projects/expo-ft || exit 1
mkdir -p logs/slurm

ckpt_of()  { case "$1" in clean) echo "$CLEAN" ;; noisy) echo "$NOISY" ;; teleop) echo "$TELEOP" ;; esac; }
demos_of() { case "$1" in clean) echo demos/isaaclab/ForgePegInsert_Robotiq_scripted ;; noisy) echo demos/isaaclab/ForgePegInsert_Robotiq_noisy ;;
                          teleop) echo demos/isaaclab/ForgePegInsert_Robotiq_teleop ;; esac; }

submit() {   # submit <run> <sft kind> <offline ratio>
    local run="$1" kind="$2" ratio="$3" ckpt; ckpt=$(ckpt_of "$kind")
    [ -n "$ckpt" ] || { echo "ERROR: --${kind}-ckpt is needed for the run $run" >&2; return 1; }
    local id
    id=$(sbatch --parsable --job-name="forge_rl_$run" --time="${HOURS}:00:00" jobs/job_forge_rl.sh \
        --sft-checkpoint "$ckpt" --demos-dir "$(demos_of "$kind")" --offline-ratio "$ratio" --tag "$run" \
        ${SETS[@]+"${SETS[@]}"}) || { echo "ERROR: sbatch failed for $run" >&2; return 1; }
    echo "$run: job ${id%%;*}  (SFT $kind, offline_ratio $ratio, demos $(demos_of "$kind"))  -> logs/slurm/forge_rl_${run}_${id%%;*}.out"
}

# check everything BEFORE submitting anything
for run in ${ONLY//,/ }; do
    case "$run" in
        nodemo) kind=$FROM ;; clean|noisy|teleop) kind=$run ;;
        *) echo "ERROR: unknown run $run in --only" >&2; exit 1 ;;
    esac
    c=$(ckpt_of "$kind"); [ -n "$c" ] || { echo "ERROR: --${kind}-ckpt is needed for the run $run" >&2; exit 1; }
    [ -d "$c/params" ] || { echo "ERROR: $c/params not found (give the checkpoint STEP directory)" >&2; exit 1; }
done
for run in ${ONLY//,/ }; do
    case "$run" in
        nodemo) submit nodemo "$FROM" 0.0 ;;
        clean|noisy|teleop) submit "$run" "$run" 0.5 ;;
    esac
done
echo "squeue -u \$USER to follow; scancel <job> stops one run (server and client are in the same job)."
