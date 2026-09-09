"""
Compare eval_rigorous performance WITH vs WITHOUT the residual policy, across
every checkpoint of an already-finished RL run -- i.e. does the trained critic
choosing among raw (un-edited) VLA samples alone explain most of the
improvement over the frozen SFT baseline, or does the residual correction add
real value on top of that selection? Same question as
train_pi_robo.py's eval_rigorous vs eval_rigorous_critic_only metrics
(logged live during training) -- this script is for running the same
comparison after the fact, on checkpoints from a run that predates that
logging, or on a run you don't want to resume just to get these numbers.

For each checkpoint step:
  - full:        eval_policy.py --rl-checkpoint --deterministic
                 (residual correction + critic selection, matches
                 train_pi_robo.py's eval_rigorous/success_rate)
  - critic_only: same, plus --critic-only (n_edit_samples=0 -- critic still
                 picks among N raw VLA samples, residual never invoked;
                 matches eval_rigorous_critic_only/success_rate)

Both variants use the EXACT same fixed episode seeds per checkpoint (one
shared seed list for the whole sweep), so the comparison isn't confounded by
different object/goal draws.

Usage:
    python scripts/eval_curve_critic_only.py \
        --config configs/task/maniskill/stack_cube.yaml \
        --checkpoints-dir logs/stack_cube/<run>/checkpoints \
        --n-episodes 200

Outputs (written into --output-dir, default = --checkpoints-dir):
    episode_seeds.json               the fixed seed list (shared by both curves)
    results/full/<step>.json         full-pipeline per-checkpoint result
    results/critic_only/<step>.json  critic-only per-checkpoint result
    comparison.json                  aggregated {step, full_sr, critic_only_sr, ...}
    comparison.png / .pdf            both curves on one plot
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_curve import discover_checkpoints, get_or_create_episode_seeds  # noqa: E402


def run_one_eval(config_path, checkpoint_path, seeds_path, output_json, log_path,
                  deterministic=False, critic_only=False):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_policy.py"),
        "--config", str(config_path),
        "--rl-checkpoint", str(checkpoint_path),
        "--episode-seeds", str(seeds_path),
        "--output-json", str(output_json),
        "--no-video",
    ]
    if deterministic:
        cmd.append("--deterministic")
    if critic_only:
        cmd.append("--critic-only")

    print(f"\n$ {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_path}")
        return False
    return True


def rebuild_comparison(results_dir: Path, comparison_json_path: Path, comparison_png_path: Path, task_label: str):
    def load_variant(subdir):
        entries = []
        variant_dir = results_dir / subdir
        if not variant_dir.exists():
            return entries
        for result_path in variant_dir.glob("*.json"):
            with open(result_path) as f:
                data = json.load(f)
            label = result_path.stem
            step = 0 if label == "base" else int(label)
            successes = data.get("successes")
            if successes:
                n = len(successes)
                p = data["success_rate"]
                se = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
            else:
                se = None
            entries.append({
                "step": step,
                "success_rate": data["success_rate"],
                "success_se": se,
                "n_episodes": data["n_episodes"],
            })
        entries.sort(key=lambda e: e["step"])
        return entries

    full_entries = load_variant("full")
    critic_only_entries = load_variant("critic_only")

    comparison = {"full": full_entries, "critic_only": critic_only_entries}
    with open(comparison_json_path, "w") as f:
        json.dump(comparison, f, indent=2)

    if not full_entries and not critic_only_entries:
        return comparison

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    def plot_variant(entries, label, color):
        if not entries:
            return
        steps = [e["step"] for e in entries]
        rates = [e["success_rate"] * 100 for e in entries]
        errs = [(e["success_se"] * 100 if e["success_se"] is not None else 0.0) for e in entries]
        ax.errorbar(
            steps, rates, yerr=errs, marker="o", markersize=7, linewidth=3,
            capsize=5, elinewidth=1.5, ecolor="black", alpha=0.9, color=color,
            label=label,
        )
        for s, r in zip(steps, rates):
            ax.annotate(f"{r:.0f}%", (s, r), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=10, weight='bold', color=color)

    plot_variant(full_entries, "Full pipeline (residual + selection)", "#1f77b4")
    plot_variant(critic_only_entries, "Critic-only (selection, no residual)", "#d62728")

    ax.set_xlabel("RL training step", labelpad=10)
    ax.set_ylabel("Success rate (%)  \u00b1 1 SE", labelpad=10)
    ax.set_title(f"Full pipeline vs. critic-only selection — {task_label}", pad=15, weight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(comparison_png_path, dpi=200)
    comparison_pdf_path = comparison_png_path.with_suffix('.pdf')
    fig.savefig(comparison_pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved highly-readable vector plot to: {comparison_pdf_path}")
    plt.close(fig)

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoints-dir", required=True,
        help="An RL run's checkpoints/ directory (numeric step subfolders). "
             "SFT checkpoints don't have a trained critic, so this script "
             "only makes sense for RL/EXPOLearner checkpoints.",
    )
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write episode_seeds.json / results/ / comparison.json / "
             "comparison.png. Defaults to --checkpoints-dir.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master seed for the fixed episode list.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run evaluation even for checkpoints that already have a result file.",
    )
    parser.add_argument(
        "--start-checkpoint", default=None,
        help="Path to the SFT checkpoint step dir the RL run started from "
             "(e.g. .../sft/.../<step>). Evaluated once as the step=0 point "
             "on the 'full' curve only (no critic to speak of at the SFT "
             "starting point, so it's not meaningful for critic_only).",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoints_dir
    results_dir = output_dir / "results"
    (results_dir / "full").mkdir(parents=True, exist_ok=True)
    (results_dir / "critic_only").mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    comparison_json_path = output_dir / "comparison.json"
    comparison_png_path = output_dir / "comparison.png"
    task_label = Path(args.config).stem

    episode_seeds = get_or_create_episode_seeds(output_dir, args.n_episodes, args.seed)
    seeds_path = output_dir / "episode_seeds.json"

    def maybe_run(subdir, label, checkpoint_path, deterministic, critic_only):
        output_json = results_dir / subdir / f"{label}.json"
        if output_json.exists() and not args.force:
            print(f"Skipping {subdir}/{label} (already evaluated; use --force to re-run)")
            return
        ok = run_one_eval(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            seeds_path=seeds_path,
            output_json=output_json,
            log_path=logs_dir / f"{subdir}_{label}.log",
            deterministic=deterministic,
            critic_only=critic_only,
        )
        if ok:
            rebuild_comparison(results_dir, comparison_json_path, comparison_png_path, task_label)

    # 'full' curve, step 0 reference point (no critic_only equivalent -- see --start-checkpoint help)
    if args.start_checkpoint is not None:
        maybe_run("full", "base", args.start_checkpoint, deterministic=False, critic_only=False)

    steps = discover_checkpoints(checkpoints_dir)
    for step in steps:
        ckpt = checkpoints_dir / str(step)
        maybe_run("full", str(step), ckpt, deterministic=True, critic_only=False)
        maybe_run("critic_only", str(step), ckpt, deterministic=True, critic_only=True)

    comparison = rebuild_comparison(results_dir, comparison_json_path, comparison_png_path, task_label)
    n_full = len(comparison["full"])
    n_critic_only = len(comparison["critic_only"])
    print(f"\nDone: {n_full} full-pipeline point(s), {n_critic_only} critic-only point(s). "
          f"See {comparison_png_path}")


if __name__ == "__main__":
    main()
