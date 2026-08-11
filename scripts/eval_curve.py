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


def run_one_eval(config_path, checkpoint_path, seeds_path, output_json, log_path, save_videos=False, is_rl_checkpoint=False, video_dir=None, deterministic=False):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_policy.py"),
        "--config", str(config_path),
        "--episode-seeds", str(seeds_path),
        "--output-json", str(output_json),
    ]
    if not save_videos:
        cmd.append("--no-video")
    elif video_dir is not None:
        cmd += ["--video-dir", str(video_dir)]
    if checkpoint_path is not None:
        flag = "--rl-checkpoint" if is_rl_checkpoint else "--checkpoint"
        cmd += [flag, str(checkpoint_path)]
    if deterministic:
        cmd.append("--deterministic")

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

        # Standard error of a proportion (successes are 0/1 per episode):
        # SE = sqrt(p*(1-p)/n). Computed from the per-episode successes already
        # saved by eval_policy.py — no need to re-run anything.
        successes = data.get("successes")
        if successes:
            n = len(successes)
            p = data["success_rate"]
            se = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
        else:
            se = None  # older result files saved before "successes" was recorded

        entries.append({
            "label": label,
            "step": step,
            "success_rate": data["success_rate"],
            "success_se": se,
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

    # Ajustement global pour la visibilité des axes et textes du rapport
    plt.rcParams.update({
        'font.size': 14,          
        'axes.titlesize': 15,     
        'axes.labelsize': 14,     
        'xtick.labelsize': 12,    
        'ytick.labelsize': 12,    
    })

    steps = [e["step"] for e in entries]
    rates = [e["success_rate"] * 100 for e in entries]
    errs = [(e["success_se"] * 100 if e["success_se"] is not None else 0.0) for e in entries]

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Courbe épaissie (linewidth=3) et marqueurs plus gros (markersize=7)
    ax.errorbar(
        steps, rates, yerr=errs, marker="o", markersize=7, linewidth=3,
        capsize=5, elinewidth=1.5, ecolor="black", alpha=0.9, color="#1f77b4"
    )
    
    has_base = any(e["label"] == "base" for e in entries)
    rl_curve = any(e["label"] not in ("base",) and e["step"] > 0 for e in entries)
    if rl_curve:
        ax.set_xlabel("RL training step (0 = SFT start checkpoint)", labelpad=10)
    else:
        ax.set_xlabel("SFT training iteration (0 = base model)", labelpad=10)
        
    ax.set_ylabel("Success rate (%)  \u00b1 1 SE", labelpad=10)
    ax.set_title(f"Eval success rate vs. checkpoint — {task_label}", pad=15, weight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Annotations agrandies (fontsize=10) et mises en gras
    for e, s, r in zip(entries, steps, rates):
        ax.annotate(f"{r:.0f}%", (s, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, weight='bold')
        
    fig.tight_layout()
    
    # Sauvegarde de la version PNG classique haute résolution (dpi=200)
    fig.savefig(curve_png_path, dpi=200)
    
    # Génération et sauvegarde automatique de la version vectorielle PDF (équivalent SVG, optimal pour LaTeX)
    curve_pdf_path = curve_png_path.with_suffix('.pdf')
    fig.savefig(curve_pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved highly-readable vector plot to: {curve_pdf_path}")
    
    plt.close(fig)

    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoints-dir", required=True,
        help="Directory containing numeric checkpoint subfolders, e.g. "
             "logs/stack_cube/<run>/sft/<sft_config_name>/<sft_exp_name>/ for an "
             "SFT sweep, or <run>/checkpoints for an RL sweep (--rl-curve).",
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
        help="Skip baseline evaluation.",
    )
    parser.add_argument(
        "--rl-curve", action="store_true",
        help="Evaluate RL/EXPOLearner checkpoints (via eval_policy.py's "
             "--rl-checkpoint) instead of SFT/openpi-style ones. "
             "--checkpoints-dir should then point at an RL run's own "
             "checkpoints/ directory (numeric step subfolders).",
    )
    parser.add_argument(
        "--start-checkpoint", default=None,
        help="With --rl-curve: path to the SFT checkpoint step dir the RL run "
             "started from (e.g. .../sft/.../<step>). Evaluated once as the "
             "step=0 point, so the RL curve reads continuously from its actual "
             "starting point instead of jumping in cold at the first RL "
             "checkpoint. Ignored without --rl-curve.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Evaluate RL checkpoints with the residual actor's mode instead of "
             "a stochastic sample (passed through to eval_policy.py). Only "
             "applies to RL checkpoints, which have a trained residual to be "
             "deterministic about -- ignored for --start-checkpoint/baseline SFT "
             "evaluation, which doesn't.",
    )
    parser.add_argument(
        "--save-videos", action="store_true",
        help="Save rollout videos for every evaluated checkpoint. Off by "
             "default: a sweep over many checkpoints would otherwise write "
             "hundreds of videos.",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoints_dir
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    curve_json_path = output_dir / "curve.json"
    curve_png_path = output_dir / "curve.png"
    task_label = Path(args.config).stem

    episode_seeds = get_or_create_episode_seeds(output_dir, args.n_episodes, args.seed)
    seeds_path = output_dir / "episode_seeds.json"

    def maybe_run(label, checkpoint_path, is_rl_checkpoint, deterministic):
        output_json = results_dir / f"{label}.json"
        if output_json.exists() and not args.force:
            print(f"Skipping {label} (already evaluated; use --force to re-run)")
            return
        ok = run_one_eval(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            seeds_path=seeds_path,
            output_json=output_json,
            log_path=logs_dir / f"{label}.log",
            save_videos=args.save_videos,
            is_rl_checkpoint=is_rl_checkpoint,
            deterministic=deterministic,
        )
        # Rebuilt after EVERY checkpoint, not just once at the end, per this
        # file's own docstring: "a sweep that gets killed partway still
        # leaves an up-to-date plot on disk."
        if ok:
            rebuild_curve(results_dir, curve_json_path, curve_png_path, task_label)

    if args.rl_curve:
        if args.start_checkpoint is not None and not args.skip_base:
            maybe_run("base", args.start_checkpoint, is_rl_checkpoint=False, deterministic=False)
        for step in discover_checkpoints(checkpoints_dir):
            maybe_run(str(step), checkpoints_dir / str(step), is_rl_checkpoint=True, deterministic=args.deterministic)
    else:
        if not args.skip_base:
            maybe_run("base", None, is_rl_checkpoint=False, deterministic=False)
        for step in discover_checkpoints(checkpoints_dir):
            maybe_run(str(step), checkpoints_dir / str(step), is_rl_checkpoint=False, deterministic=False)

    entries = rebuild_curve(results_dir, curve_json_path, curve_png_path, task_label)
    print(f"\nDone: {len(entries)} points on the curve. See {curve_png_path}")


if __name__ == "__main__":
    main()
