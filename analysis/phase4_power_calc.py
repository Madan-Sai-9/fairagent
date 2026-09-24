"""
analysis/phase4_power_calc.py

Reads Phase 4's pilot per-client results (results/phase4_pilot_clients.csv)
and, per axis:
  1. fits a logistic regression selected ~ variant + true-utility covariates
     + list_position, clustered by trial_id — the same model shape Phase 6
     runs on the full Core matrix — restricted to LLM-strategy rows (the
     numeric-control selector's rows are the negative control, not part of
     this calculation: they should show ~0 variant effect by construction,
     since numeric selectors never read metadata_text),
  2. converts the pilot's observed variant odds ratio into the per-cell
     trial count Phase 5's Core matrix needs for 80% power at alpha=0.05.

This is deliberately a two-proportion-test approximation of the fitted
effect, not a full power simulation of the mixed-effects model itself — a
standard, conservative simplification for setting a trial-count TARGET this
early; Phase 6 runs the real model on the full matrix.

Usage (from a Kaggle notebook or local machine with the pilot CSV):
    python analysis/phase4_power_calc.py results/phase4_pilot_clients.csv
"""

from __future__ import annotations

import math
import sys

NUMERIC_CONTROL_NAMES = {"power_of_choice", "oort", "random"}
COVARIATES = [
    "dataset_size", "historical_accuracy", "historical_loss",
    "bandwidth_mbps", "rounds_participated", "list_position",
]


def load_llm_rows(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df[~df["selector"].isin(NUMERIC_CONTROL_NAMES)].copy()


def fit_variant_effect(df):
    """Fits one clustered logistic regression per axis subset `df`.
    Returns (odds_ratio, p_control, n_obs, statsmodels_result)."""
    import statsmodels.formula.api as smf

    variants = sorted(df["variant"].unique())
    if len(variants) != 2:
        raise ValueError(f"Expected exactly 2 variants in this axis's data, got {variants}")
    reference, treatment = variants
    df = df.copy()
    df["variant_bin"] = (df["variant"] == treatment).astype(int)

    formula = "selected ~ variant_bin + " + " + ".join(COVARIATES)
    fit = smf.logit(formula, data=df).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": df["trial_id"]}
    )

    odds_ratio = math.exp(fit.params["variant_bin"])
    p_control = df.loc[df["variant_bin"] == 0, "selected"].mean()
    return odds_ratio, p_control, len(df), fit, reference, treatment


def required_trials_per_cell(p_control: float, odds_ratio: float,
                              clients_per_trial_row: float,
                              alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-proportion z-test sample size (per arm, in CLIENT rows) needed to
    detect the pilot's observed odds ratio, converted to a per-cell TRIAL
    count by dividing by how many client rows one trial contributes to each
    arm on average (~pool_size / 2, since variant assignment is a coin flip)."""
    from scipy.stats import norm

    p1 = min(max(p_control, 1e-6), 1 - 1e-6)
    odds1 = p1 / (1 - p1)
    odds2 = odds1 * odds_ratio
    p2 = odds2 / (1 + odds2)

    if p1 == p2:
        return float("inf")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n_rows_per_arm = numerator / (p1 - p2) ** 2
    n_trials = n_rows_per_arm / (clients_per_trial_row / 2)
    return math.ceil(n_trials)


def main(csv_path: str, pool_size: int = 10):
    df = load_llm_rows(csv_path)
    axes = sorted(df["axis"].unique())

    lines = ["# Phase 4 Pilot — Power Calculation\n", f"Source: `{csv_path}`\n"]
    for axis in axes:
        sub = df[df["axis"] == axis]
        odds_ratio, p_control, n_obs, fit, ref, treat = fit_variant_effect(sub)
        n_trials = required_trials_per_cell(p_control, odds_ratio, clients_per_trial_row=pool_size)

        print(f"\n=== axis: {axis} ({n_obs} client-selector rows) ===")
        print(f"  reference variant: {ref!r}, treatment variant: {treat!r}")
        print(f"  P(selected | reference, at covariate means): {p_control:.4f}")
        print(f"  variant odds ratio (treatment vs reference): {odds_ratio:.4f}")
        print(f"  variant coefficient p-value: {fit.pvalues['variant_bin']:.4g}")
        print(f"  -> Phase 5 target: ~{n_trials} trials per (strategy, axis) cell "
              f"for 80% power at alpha=0.05")

        lines.append(f"\n## Axis: `{axis}`\n")
        lines.append(f"- Pilot rows: {n_obs} (LLM-strategy client-selector rows only)\n")
        lines.append(f"- Reference variant: `{ref}`, treatment variant: `{treat}`\n")
        lines.append(f"- P(selected | reference): {p_control:.4f}\n")
        lines.append(f"- Variant odds ratio: {odds_ratio:.4f} (p={fit.pvalues['variant_bin']:.4g})\n")
        lines.append(f"- **Required trials per (strategy, axis) cell for Phase 5: ~{n_trials}**\n")

    out_path = "analysis/phase4_power_calc_results.md"
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python analysis/phase4_power_calc.py <path-to-phase4_pilot_clients.csv>")
    main(sys.argv[1])
