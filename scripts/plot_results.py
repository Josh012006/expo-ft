#!/usr/bin/env python3
"""
Standalone script to regenerate highly-readable plots (PNG and PDF)
directly from an existing 'results/' directory containing checkpoint JSON files.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from a results directory.")
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Path to the 'results' folder containing the checkpoint .json files."
    )
    parser.add_argument(
        "--task-label", default="Task", type=str,
        help="Label of the task to display in the plot title (e.g., 'StackCube-v1')."
    )
    parser.add_argument(
        "--output-dir", default=None, type=Path,
        help="Where to save curve.png and curve.pdf. Defaults to the parent of results-dir."
    )
    args = parser.parse_args()

    if not args.results_dir.exists() or not args.results_dir.is_dir():
        print(f"Error: The directory {args.results_dir} does not exist.")
        return

    # 1. Extraction et lecture des fichiers JSON
    entries = []
    for result_path in args.results_dir.glob("*.json"):
        with open(result_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {result_path}, skipping.")
                continue
                
        label = result_path.stem
        if label == "curve": # Ignore un éventuel curve.json déjà présent
            continue
            
        step = 0 if label == "base" else int(label)

        successes = data.get("successes")
        if successes:
            n = len(successes)
            p = data["success_rate"]
            se = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
        else:
            se = data.get("success_se", 0.0) # Fallback si déjà calculé

        entries.append({
            "label": label,
            "step": step,
            "success_rate": data["success_rate"],
            "success_se": se,
        })
        
    entries.sort(key=lambda e: e["step"])

    if not entries:
        print(f"No valid checkpoint JSON files found in {args.results_dir}.")
        return

    # 2. Configuration graphique Matplotlib (Haute Visibilité Académique)
    plt.rcParams.update({
        'font.size': 14,          
        'axes.titlesize': 15,     
        'axes.labelsize': 14,     
        'xtick.labelsize': 12,    
        'ytick.labelsize': 12,    
    })

    steps = [e["step"] for e in entries]
    rates = [e["success_rate"] * 100 for e in entries]
    errs = [e["success_se"] * 100 for e in entries]

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Courbe épaisse et marqueurs visibles
    ax.errorbar(
        steps, rates, yerr=errs, marker="o", markersize=7, linewidth=3,
        capsize=5, elinewidth=1.5, ecolor="black", alpha=0.9, color="#1f77b4"
    )
    
    # Détection automatique du type de courbe pour l'axe X
    rl_curve = any(e["label"] not in ("base",) and e["step"] > 0 for e in entries)
    if rl_curve:
        ax.set_xlabel("RL training step (0 = SFT start checkpoint)", labelpad=10)
    else:
        ax.set_xlabel("SFT training iteration (0 = base model)", labelpad=10)
        
    ax.set_ylabel("Success rate (%)  \u00b1 1 SE", labelpad=10)
    ax.set_title(f"Eval success rate vs. checkpoint — {args.task_label}", pad=15, weight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Écritures des pourcentages en gros et gras au-dessus des points
    for s, r in zip(steps, rates):
        ax.annotate(f"{r:.0f}%", (s, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, weight='bold')
        
    fig.tight_layout()
    
    # 3. Sauvegarde des fichiers
    out_dir = args.output_dir if args.output_dir else args.results_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "curve.png"
    pdf_path = out_dir / "curve.pdf"
    
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    
    print(f"\nSuccessfully generated plots in: {out_dir}")
    print(f"  -> PNG: {png_path.name}")
    print(f"  -> PDF (Vector): {pdf_path.name}")
    plt.close(fig)

if __name__ == "__main__":
    main()
