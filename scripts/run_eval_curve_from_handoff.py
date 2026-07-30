"""
Reads the handoff file train_pi_robo.py writes just before exiting, and runs
the 200-fixed-seed eval_curve sweep + wandb logging -- as a genuinely
SEPARATE process, invoked by job_rl.sh AFTER train_pi_robo.py's own process
has fully exited.

Why this needs to be a separate invocation rather than a function call made
from inside train_pi_robo.py itself: XLA_PYTHON_CLIENT_MEM_FRACTION
preallocation is held for a JAX process's entire lifetime, not released just
because its training loop has finished. Calling eval_curve.py via subprocess
while train_pi_robo.py's own process is still alive (blocked waiting on it)
starves every eval subprocess of GPU memory -- observed directly: every
per-checkpoint eval OOM'd trying to allocate ~1GB more, with the training
process still shown holding ~78GB in the same nvidia-smi accounting report.

Usage (see job_rl.sh):
    python scripts/run_eval_curve_from_handoff.py \
        --handoff logs/eval_curve_handoff_${SLURM_JOB_ID}.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import wandb

from expo_ft.utils.eval_curve_runner import run_eval_curve, log_eval_curve_to_wandb


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", required=True,
                         help="Path to the eval_curve_handoff_<job_id>.json written by train_pi_robo.py")
    args = parser.parse_args()

    handoff_path = Path(args.handoff)
    if not handoff_path.exists():
        logging.error(f"[eval_curve] Handoff file not found: {handoff_path} -- "
                       f"eval_curve_enabled was probably false for this run, or "
                       f"train_pi_robo.py exited before reaching the write. Nothing to do.")
        return

    with open(handoff_path) as f:
        info = json.load(f)

    ok = run_eval_curve(
        config_path=info["task_config"],
        checkpoints_dir=info["checkpoints_dir"],
        n_episodes=info["n_episodes"],
        start_checkpoint=info["start_checkpoint"],
    )
    if not ok:
        logging.error("[eval_curve] Sweep failed -- see per-checkpoint logs under "
                       "<checkpoints_dir>/logs/. Skipping wandb logging.")
        return

    # Reattach to the SAME wandb run training logged everything else to
    # (not a new run) -- resume="must" fails loudly if the run ID is wrong
    # rather than silently creating an unrelated new run.
    wandb.init(project=info["wandb_project"], id=info["wandb_run_id"], resume="must")
    log_eval_curve_to_wandb(info["checkpoints_dir"])
    wandb.finish()


if __name__ == "__main__":
    main()
