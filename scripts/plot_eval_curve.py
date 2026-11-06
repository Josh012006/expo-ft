"""Rebuild the plot of an eval-curve run from its results, without evaluating anything (no server, no GPU, no checkpoint).

Reads <eval_dir>/results/*.json (base.json, 20000.json, ... written by scripts/eval_curve.py) and writes the curve:
curve.png, curve.pdf and curve.json (or the files named by --output). Same figure as scripts/eval_curve.py, with the title
and the x-axis you choose. Use it to give a title to a curve evaluated earlier, or to redo a figure after a change.

Usage:
    python scripts/plot_eval_curve.py <eval_dir> --rl-curve --title "Teleop demos, offline ratio 0.5"
    python scripts/plot_eval_curve.py logs/forge_peg_insert/<run>_rl/eval_curve --rl-curve --title "Clean" \
        --output logs/figures/curve_clean.png

    <eval_dir>   the --output-dir of the eval-curve job (the folder that contains results/)
    --rl-curve   RL curve: x-axis "RL training step (0 = SFT start checkpoint)" (default: "SFT training iteration")
    --title      full title of the figure (default: "Eval success rate vs. checkpoint — <name of the run folder>")
    --output     PNG to write (default: <eval_dir>/curve.png); the .pdf and the .json go next to it
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_curve import rebuild_curve  # noqa: E402  (the very same figure as the eval job)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", help="folder that contains results/ (the --output-dir of the eval-curve job)")
    parser.add_argument("--title", default=None, help="full title of the figure")
    parser.add_argument("--rl-curve", action="store_true", help="x-axis of an RL curve instead of an SFT curve")
    parser.add_argument("--output", default=None, help="PNG to write (default: <eval_dir>/curve.png)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    results_dir = eval_dir / "results"
    if not results_dir.is_dir() or not any(results_dir.glob("*.json")):
        sys.exit(f"ERROR: no results/*.json in {eval_dir}")

    png = Path(args.output).resolve() if args.output else eval_dir / "curve.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    entries = rebuild_curve(
        results_dir, png.with_suffix(".json"), png, task_label=eval_dir.parent.name,
        rl_curve=args.rl_curve, title=args.title,
    )
    print(f"{len(entries)} points: " + ", ".join(f"{e['label']}={e['success_rate'] * 100:.0f}%" for e in entries))
    print(f"Written: {png} (+ .pdf, .json)")


if __name__ == "__main__":
    main()
