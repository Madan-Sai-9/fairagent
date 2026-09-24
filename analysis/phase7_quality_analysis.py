"""
analysis/phase7_quality_analysis.py

Phase 7 quality trade-off analysis: compares final FedAvg test accuracy
between the original phrased metadata and Phase 7's neutral-templating
mitigation, for each (strategy, axis) cell run by
experiments/run_phase7_quality_check.py. Answers: does removing the
phrasing bias cost anything in actual model performance?

Writes:
  - analysis/phase7_quality_results.md   (final-accuracy table + deltas)
  - analysis/phase7_quality_curves.png   (accuracy-vs-round, all 4 runs)

Usage:
    python analysis/phase7_quality_analysis.py results/phase7_quality_check.csv
"""

from __future__ import annotations

import sys

import pandas as pd


def analyze(csv_path: str):
    df = pd.read_csv(csv_path)
    final = df.sort_values("round").groupby("run_name").tail(1)

    lines = ["# Phase 7 — Quality Trade-off Check\n", f"\nSource: `{csv_path}`\n"]
    lines.append("\n| Run | Strategy | Axis | Condition | Final test accuracy |\n")
    lines.append("|---|---|---|---|---|\n")
    for _, r in final.sort_values(["strategy", "axis", "condition"]).iterrows():
        lines.append(f"| {r['run_name']} | {r['strategy']} | {r['axis']} | {r['condition']} | {r['test_accuracy']:.4f} |\n")

    lines.append("\n## Phrased vs. neutral delta\n\n")
    lines.append("| Strategy | Axis | Phrased | Neutral | Delta (neutral - phrased) |\n")
    lines.append("|---|---|---|---|---|\n")
    for (strategy, axis), group in final.groupby(["strategy", "axis"]):
        phrased = group[group["condition"] == "phrased"]["test_accuracy"]
        neutral = group[group["condition"] == "neutral"]["test_accuracy"]
        if len(phrased) == 1 and len(neutral) == 1:
            p, n = phrased.iloc[0], neutral.iloc[0]
            lines.append(f"| {strategy} | {axis} | {p:.4f} | {n:.4f} | {n - p:+.4f} |\n")

    results_md_path = "analysis/phase7_quality_results.md"
    with open(results_md_path, "w") as f:
        f.writelines(lines)
    print("Written:", results_md_path)

    plot_curves(df)


def plot_curves(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, group in df.groupby("run_name"):
        group = group.sort_values("round")
        ax.plot(group["round"], group["test_accuracy"], label=run_name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Phase 7 quality trade-off: FedAvg accuracy, phrased vs. neutral metadata")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = "analysis/phase7_quality_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Written:", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python analysis/phase7_quality_analysis.py <path-to-phase7_quality_check.csv>")
    analyze(sys.argv[1])
