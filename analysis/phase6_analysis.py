"""
analysis/phase6_analysis.py

Phase 6: Statistical Analysis on the completed Phase 5 Core matrix
(results/phase4_pilot_clients.csv — 1050 trials x 3 LLM strategies x 2
axes, trial-count targets fixed in advance by Phase 4's pilot power calc;
see analysis/phase4_notes.md's methodological note on not testing early).

For each (axis, selector) cell — the 3 LLM strategies plus the numeric
control, on identical client pools — fits:
  1. a chi-square test of independence: variant x selected (unconditional,
     a simple first check, ignores covariates/clustering),
  2. a cluster-robust logistic regression: selected ~ variant +
     true-utility covariates + list_position, clustered by trial_id (same
     model shape as Phase 4's pilot power calc). statsmodels has no
     turnkey GLMM for a binary DV at this scale, so cluster-robust SEs on
     the fixed-effects logit — clustering on the by-trial grouping a true
     random intercept would model — is the standard substitute here.

The paper's central causal contrast is then read directly off this table:
an LLM strategy showing a variant effect while the numeric control shows
none, on the SAME pools, is direct evidence the LLM orchestrator is
reading phrasing rather than only utility — the control never reads
`metadata_text`, so any effect there would be a construct-validity
failure in the harness, not a substantive finding.

Writes:
  - analysis/phase6_results.csv        (one row per (axis, selector) cell)
  - analysis/phase6_results.md         (readable summary + comparison)
  - analysis/phase6_forest_<axis>.png  (odds ratios w/ 95% CI, per selector)

Usage:
    python analysis/phase6_analysis.py results/phase4_pilot_clients.csv
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

COVARIATES = [
    "dataset_size", "historical_accuracy", "historical_loss",
    "bandwidth_mbps", "rounds_participated", "list_position",
]


def chi_square_cell(df):
    from scipy.stats import chi2_contingency
    table = pd.crosstab(df["variant"], df["selected"])
    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p


def fit_variant_effect(df):
    import statsmodels.formula.api as smf

    variants = sorted(df["variant"].unique())
    if len(variants) != 2:
        raise ValueError(f"Expected exactly 2 variants, got {variants}")
    reference, treatment = variants
    df = df.copy()
    df["variant_bin"] = (df["variant"] == treatment).astype(int)

    formula = "selected ~ variant_bin + " + " + ".join(COVARIATES)
    fit = smf.logit(formula, data=df).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": df["trial_id"]}
    )
    coef = fit.params["variant_bin"]
    se = fit.bse["variant_bin"]
    ci_low, ci_high = coef - 1.96 * se, coef + 1.96 * se
    return {
        "reference": reference,
        "treatment": treatment,
        "odds_ratio": math.exp(coef),
        "or_ci_low": math.exp(ci_low),
        "or_ci_high": math.exp(ci_high),
        "p_value": fit.pvalues["variant_bin"],
        "n_rows": len(df),
        "n_trials": df["trial_id"].nunique(),
    }


def analyze(csv_path: str):
    df = pd.read_csv(csv_path)
    selectors = sorted(df["selector"].unique())
    axes = sorted(df["axis"].unique())

    rows = []
    for axis in axes:
        for selector in selectors:
            cell = df[(df["axis"] == axis) & (df["selector"] == selector)]
            if cell.empty or cell["variant"].nunique() != 2:
                continue
            chi2, chi_p = chi_square_cell(cell)
            reg = fit_variant_effect(cell)
            rows.append({"axis": axis, "selector": selector, "chi2": chi2, "chi2_p": chi_p, **reg})

    results = pd.DataFrame(rows)
    results.to_csv("analysis/phase6_results.csv", index=False)
    print("Written: analysis/phase6_results.csv")

    write_markdown_summary(results, csv_path)
    for axis in axes:
        plot_forest(results[results["axis"] == axis], axis)
    return results


def write_markdown_summary(results, csv_path):
    llm_selectors = [s for s in results["selector"].unique() if s.startswith("llm_")]
    control_selectors = [s for s in results["selector"].unique() if not s.startswith("llm_")]

    lines = ["# Phase 6 — Statistical Analysis\n", f"\nSource: `{csv_path}`\n"]
    for axis in sorted(results["axis"].unique()):
        sub = results[results["axis"] == axis]
        lines.append(f"\n## Axis: `{axis}`\n\n")
        lines.append("| Selector | Reference -> Treatment | Odds ratio (95% CI) | p (regression) | p (chi-sq) | n trials |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['selector']} | {r['reference']} -> {r['treatment']} | "
                f"{r['odds_ratio']:.3f} ({r['or_ci_low']:.3f}-{r['or_ci_high']:.3f}) | "
                f"{r['p_value']:.4g} | {r['chi2_p']:.4g} | {r['n_trials']} |\n"
            )

        llm_rows = sub[sub["selector"].isin(llm_selectors)]
        ctrl_rows = sub[sub["selector"].isin(control_selectors)]
        if not llm_rows.empty and not ctrl_rows.empty:
            ctrl = ctrl_rows.iloc[0]
            lines.append(
                f"\n**Agentic vs. numeric-control disparity:** LLM strategies' odds "
                f"ratios range {llm_rows['odds_ratio'].min():.3f}-{llm_rows['odds_ratio'].max():.3f} "
                f"on identical pools where the numeric control (`{ctrl['selector']}`) shows "
                f"OR={ctrl['odds_ratio']:.3f} (p={ctrl['p_value']:.4g}) — the control never reads "
                f"`metadata_text`, so any effect there would flag a construct-validity problem "
                f"in the harness, not evidence against the LLM finding.\n"
            )

    with open("analysis/phase6_results.md", "w") as f:
        f.writelines(lines)
    print("Written: analysis/phase6_results.md")


def plot_forest(sub, axis):
    import matplotlib.pyplot as plt

    sub = sub.sort_values("selector")
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(sub) + 1))
    y = np.arange(len(sub))
    ax.errorbar(
        sub["odds_ratio"], y,
        xerr=[sub["odds_ratio"] - sub["or_ci_low"], sub["or_ci_high"] - sub["odds_ratio"]],
        fmt="o", capsize=3,
    )
    ax.axvline(1.0, linestyle="--", color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["selector"])
    ax.set_xlabel("Odds ratio (treatment vs. reference), 95% CI")
    ax.set_title(f"Phrasing-variant effect on selection — axis: {axis}")
    fig.tight_layout()
    out_path = f"analysis/phase6_forest_{axis}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Written:", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python analysis/phase6_analysis.py <path-to-phase4_pilot_clients.csv>")
    analyze(sys.argv[1])
