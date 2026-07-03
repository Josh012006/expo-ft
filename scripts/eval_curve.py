"""
Evaluate baseline + every SFT checkpoint in a directory, on a FIXED set of episodes
(same seeds for every checkpoint, generated once and reused), and plot success rate
as a function of training iteration.

Design notes:
  - Fixed episodes across checkpoints: without this, differences between checkpoints
    would be confounded with differences in which object/goal positions env.reset()
    happened to draw. The seed list is generated once (from --seed) and cached to
    <output-dir>/episode_seeds.json so re-running the sweep later (e.g. to add newer
    checkpoints) stays comparable with earlier results.
  - Resumable: each checkpoint's result is written to disk as soon as it's evaluated
    (results/<label>.json). Re-running the script skips any checkpoint that already
    has a result file, unless --force is given. The aggregate curve.json/curve.png are
    rebuilt from whatever results exist after every single checkpoint finishes, so a
    sweep that gets killed partway still leaves an up-to-date plot on disk.
  - Each checkpoint is evaluated in its own subprocess (fresh eval_policy.py invocation)
    rather than importing evaluate() and looping in-process — avoids JAX/GPU memory not
    being fully released between sequentially loaded models, which is the usual failure
    mode when chaining several large-model evals in one long-lived process.

Usage:
    python scripts/eval_curve.py \
        --config configs/task/maniskill/stack_cube.yaml \
        --checkpoints-dir logs/stack_cube/stack_cube_expo_ft_2026-07-02_09-08-24/sft/expo_pi05_droid_lora_finetune_sft_joint_state/stack_cube_sft \
        --n-episodes 50

Outputs (written into --output-dir, default = --checkpoints-dir):
    episode_seeds.json   the fixed seed list (generated once, reused on reruns)
    results/base.json    per-run structured result (from eval_policy.py --output-json)
    results/200.json
    results/400.json
    ...
    curve.json           aggregated {step, success_rate, n_episodes} across all runs
    curve.png            success rate vs. training iteration
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def discover_checkpoints(checkpoints_dir: Path):
    """Return sorted list of int step numbers for every purely-numeric subdirectory."""
    steps = []
    for entry in checkpoints_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            steps.append(int(entry.name))
    return sorted(steps)


def get_or_create_episode_seeds(output_dir: Path, n_episodes: int, master_seed: int):
    seeds_path = output_dir / "episode_seeds.json"
    if seeds_path.exists():
        with open(seeds_path) as f:
            seeds = json.load(f)
        if len(seeds) != n_episodes:
            raise ValueError(
                f"Existing {seeds_path} has {len(seeds)} seeds but --n-episodes={n_episodes}. "
                "Either delete it to regenerate, or match --n-episodes to the existing sweep."
            )
        print(f"Reusing existing fixed episode seeds: {seeds_path}")
        return seeds

    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_episodes).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"Generated {n_episodes} fixed episode seeds (master_seed={master_seed}): {seeds_path}")
    return seeds


def run_one_eval(config_path, checkpoint_path, seeds_path, output_json, log_path):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_policy.py"),
        "--config", str(config_path),
        "--episode-seeds", str(seeds_path),
        "--output-json", str(output_json),
        "--no-video",
    ]
    if checkpoint_path is not None:
        cmd += ["--checkpoint", str(checkpoint_path)]

    print(f"\n$ {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_path}")
        return False
    return True


def rebuild_curve(results_dir: Path, curve_json_path: Path, curve_png_path: Path, task_label: str):
    entries = []
    for result_path in results_dir.glob("*.json"):
        with open(result_path) as f:
            data = json.load(f)
        label = result_path.stem
        step = 0 if label == "base" else int(label)
        entries.append({
            "label": label,
            "step": step,
            "success_rate": data["success_rate"],
            "n_episodes": data["n_episodes"],
        })
    entries.sort(key=lambda e: e["step"])

    with open(curve_json_path, "w") as f:
        json.dump(entries, f, indent=2)

    if not entries:
        return entries

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [e["step"] for e in entries]
    rates = [e["success_rate"] * 100 for e in entries]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, rates, marker="o", linewidth=2)
    ax.set_xlabel("SFT training iteration (0 = base model)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Eval success rate vs. checkpoint — {task_label}")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    for s, r in zip(steps, rates):
        ax.annotate(f"{r:.0f}%", (s, r), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(curve_png_path, dpi=150)
    plt.close(fig)

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoints-dir", required=True,
        help="Directory containing numeric checkpoint subfolders, e.g. "
             "logs/stack_cube/<run>/sft/<sft_config_name>/<sft_exp_name>/",
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write episode_seeds.json / results/ / curve.json / curve.png. "
             "Defaults to --checkpoints-dir.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master seed for the fixed episode list.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run evaluation even for checkpoints that already have a result file.",
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip the base-model (step 0) evaluation, e.g. if it was already run separately.",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoints_dir
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps = discover_checkpoints(checkpoints_dir)
    if not steps:
        raise RuntimeError(f"No numeric checkpoint subfolders found in {checkpoints_dir}")
    print(f"Found {len(steps)} checkpoints: {steps}")

    seeds = get_or_create_episode_seeds(output_dir, args.n_episodes, args.seed)
    seeds_path = output_dir / "episode_seeds.json"

    plan = []
    if not args.skip_base:
        plan.append(("base", None))
    for step in steps:
        plan.append((str(step), checkpoints_dir / str(step)))

    task_label = Path(args.config).stem

    for label, ckpt_path in plan:
        result_json = results_dir / f"{label}.json"
        if result_json.exists() and not args.force:
            print(f"[{label}] already evaluated, skipping (use --force to redo)")
        else:
            print(f"[{label}] evaluating on {args.n_episodes} fixed episodes...")
            ok = run_one_eval(
                config_path=args.config,
                checkpoint_path=ckpt_path,
                seeds_path=seeds_path,
                output_json=result_json,
                log_path=logs_dir / f"{label}.log",
            )
            if not ok:
                continue  # keep going with the rest of the sweep

        # Rebuild the aggregate curve after every checkpoint — always up to date on disk.
        entries = rebuild_curve(
            results_dir, output_dir / "curve.json", output_dir / "curve.png", task_label
        )
        if entries:
            last = entries[-1] if entries[-1]["label"] == label else next(
                e for e in entries if e["label"] == label
            )
            print(f"[{label}] success_rate={last['success_rate']:.1%} "
                  f"({last['n_episodes']} episodes)")

    print(f"\nDone. Aggregate results: {output_dir / 'curve.json'}")
    print(f"Plot: {output_dir / 'curve.png'}")


if __name__ == "__main__":
    main()
