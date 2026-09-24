# Multiple-Comparison Correction — Phase 6 Bias Findings

Phase 6 ran 6 hypothesis tests on the LLM strategies (3 strategies x 2
axes; the 2 numeric-control cells are a construct-validity sanity check,
not part of this discovery family, and are excluded from correction).
Reported here: Bonferroni (conservative, family-wise error rate) and
Benjamini-Hochberg (standard, false discovery rate) corrections, both at
alpha=0.05, applied to the cluster-robust logistic regression p-values
from `analysis/phase6_results.csv`.

| Axis | Strategy | Raw p | Bonferroni sig. (p < 0.00833) | BH q-value | BH sig. (q < 0.05) |
|---|---|---|---|---|---|
| Implied Geography | CoT | 3.48e-13 | **YES** | 2.09e-12 | **YES** |
| Implied Geography | Few-Shot | 0.00186 | **YES** | 0.00558 | **YES** |
| Formality | Few-Shot | 0.00292 | **YES** | 0.00584 | **YES** |
| Implied Geography | Description-Only | 0.0708 | no | 0.106 | no |
| Formality | CoT | 0.562 | no | 0.571 | no |
| Formality | Description-Only | 0.571 | no | 0.571 | no |

## Result: no change to the paper's conclusions

All three cells reported as significant in `analysis/phase6_notes.md` —
**CoT/geography** (the largest effect in the study), **Few-Shot/geography**,
and **Few-Shot/formality** — remain significant under both the
conservative Bonferroni family-wise correction and the standard
Benjamini-Hochberg FDR correction. Description-Only's geography trend
(raw p=0.0708) was already reported as non-significant before correction
and stays non-significant after. Nothing that was previously called
significant becomes non-significant, and nothing null becomes significant
— the finding is robust to the number of tests run.

This note exists so the manuscript's statistics section can cite a
correction was checked and had no material effect, rather than leaving
the 6-test multi-comparison question unaddressed.
