"""
Runs scripts/eval_curve.py synchronously, on the SAME node/GPU as training,
strictly AFTER the training loop has finished (so the GPU is free again) --
sequential, not parallel, no separate sbatch job, no concurrent-job resource
concerns to manage. Simpler and safer than submitting a job per checkpoint,
at the cost of adding the full eval sweep's own wall-clock time to the end
of the training job -- budget the job's --time allocation accordingly (e.g.
~10 min/checkpoint x however many checkpoints keep_period leaves on disk,
per the SFT eval_curve reference point). job_rl.sh currently requests
120:00:00, which comfortably covers training (~4h) + a sweep of a dozen
checkpoints (~2h) with a lot of room to spare.
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_eval_curve(config_path, checkpoints_dir, n_episodes, start_checkpoint):
    """Blocking call -- returns True on success, False if eval_curve.py itself
    exited non-zero (logged, never raises: a failed eval sweep shouldn't be
    allowed to erase an otherwise-successful training run's own results).
    Uses sys.executable, so it runs under the SAME already-activated venv as
    training -- no need to source setup_env.sh again."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_curve.py"),
        "--config", str(config_path),
        "--checkpoints-dir", str(checkpoints_dir),
        "--n-episodes", str(n_episodes),
        "--rl-curve",
    ]
    if start_checkpoint:
        cmd += ["--start-checkpoint", str(start_checkpoint)]

    logging.info(f"[eval_curve] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        logging.error(f"[eval_curve] Exited with code {proc.returncode} -- training's "
                       f"own results are unaffected, but the 200-seed eval curve is "
                       f"incomplete. Re-run scripts/eval_curve.py manually to backfill "
                       f"(already-done checkpoints are skipped automatically).")
        return False
    return True


def log_eval_curve_to_wandb(checkpoints_dir, key="eval_curve_200seeds"):
    """Reads the curve.json written by eval_curve.py and logs it as a wandb
    Table + line plot -- NOT via wandb.log(..., step=...), since these
    checkpoint steps are far behind the run's own already-logged step
    counter (training has already logged up to step=max_steps) and would be
    silently dropped or misordered by wandb's monotonic-step expectation. A
    Table-based plot sidesteps that entirely and renders as its own chart."""
    curve_path = Path(checkpoints_dir) / "curve.json"
    if not curve_path.exists():
        logging.error(f"[eval_curve] {curve_path} not found -- nothing to log.")
        return

    with open(curve_path) as f:
        entries = json.load(f)

    table = wandb.Table(columns=["step", "label", "success_rate", "success_se", "n_episodes"])
    for e in entries:
        table.add_data(e["step"], e["label"], e["success_rate"], e.get("success_se"), e["n_episodes"])

    wandb.log({
        key: wandb.plot.line(table, "step", "success_rate",
                              title="Eval success rate on 200 fixed seeds vs. RL checkpoint"),
        f"{key}_table": table,
    })
    logging.info(f"[eval_curve] Logged {len(entries)} points to wandb under '{key}'.")
