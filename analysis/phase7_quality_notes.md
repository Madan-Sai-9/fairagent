# Phase 7 — Quality Trade-off Check: Results & Notes

**Setup:** 4 full FedAvg runs (`configs/phase7_quality_check.yaml`, same
20-client Dirichlet CIFAR-10 partition and hyperparameters as Phase 1),
comparing original phrased metadata vs. Phase 7's neutral template, for
the two cells with the strongest selection bias in Phase 6:
Few-Shot/formality and CoT/implied_geography.

## Result: a small, consistent accuracy cost from the mitigation

| Strategy | Axis | Phrased | Neutral | Delta |
|---|---|---|---|---|
| CoT | implied_geography | 0.5866 | 0.5754 | **-0.0112** |
| Few-Shot | formality | 0.5906 | 0.5842 | **-0.0064** |

Both cells lose a small amount of final test accuracy under neutral
templating — roughly 0.6-1.1 percentage points. The direction is
consistent (neutral always slightly below phrased, never above), and the
training curves (`phase7_quality_curves.png`) track each other closely
for most of the run, with the phrased condition pulling slightly ahead
in the final third of training in both cases.

## Honest read

**This is not a free mitigation, and it shouldn't be reported as one.**
Removing the phrasing bias costs a small amount of real selection
quality in both cases it was tested. That is itself a legitimate,
reportable finding — a fairness/utility trade-off, not a failure of the
mitigation. Given the size of the bias removed (CoT's geography effect
was OR=0.40, the largest in the whole study) against a ~1 percentage
point accuracy cost, most reasonable framings would call that a
favorable trade — but the paper should state the cost explicitly rather
than imply the mitigation is costless.

## Important limitation — this is a single run per condition

Each cell here is **one seeded FedAvg run**, not an average over multiple
seeds. FedAvg training has real run-to-run variance from stochastic
minibatch ordering and the specific rounds sampled, even at a fixed
top-level seed (local training's DataLoader shuffling isn't separately
seeded per round here). A ~0.6-1.1 point gap is small enough that **it
cannot be distinguished from ordinary run-to-run noise without repeated
seeds** — this result shows a consistent direction across the two tested
cells, which is suggestive, but is not a statistically validated cost the
way Phase 6/7's selection-bias findings are (those had hundreds of trials
and clustered regression; this has n=1 run per condition).

**If time allows before submission:** re-run each of the 4 configs across
3-5 seeds and report a mean ± std delta, which would let this trade-off
be stated with actual statistical confidence rather than a single-run
point estimate. Given this project's Core-scope constraints, this is
reasonable to flag as a limitation/future-work item rather than block
on, given the plan's explicit framing of this check as "a handful of
runs," not a fully powered study in its own right.

## Provenance

Run in Kaggle (GPU, Llama-3.2-3B-Instruct) via
`experiments/run_phase7_quality_check.py`; analysis via
`analysis/phase7_quality_analysis.py`. This closes out Phase 7's push
checklist (both the statistical mitigation result and the quality
trade-off check are now committed).
