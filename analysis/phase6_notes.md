# Phase 6 — Statistical Analysis: Results & Notes

**Data:** the completed Phase 5 Core matrix — 1050 LLM trials (3 strategies
x 2 axes x 120/230 reps) plus a same-pool numeric-control (Power-of-Choice)
selection per trial. Full results: `phase6_results.csv` / `.md`, forest
plots: `phase6_forest_formality.png`, `phase6_forest_implied_geography.png`.

## Headline result

**The phrasing bias is real, but it is not uniform across prompting
strategies — it is strategy- and axis-specific.** On identical
true-utility pools, the numeric control shows no significant phrasing
effect on either axis (formality OR=0.815, p=0.082; implied geography
OR=1.013, p=0.879) — exactly the null result construct validity requires,
since Power-of-Choice never reads `metadata_text`. Against that clean
baseline, three of six LLM (strategy x axis) cells show a significant
phrasing effect, one shows a non-significant trend in the same direction,
and two show no effect:

| Axis | Strategy | Odds ratio | p | Verdict |
|---|---|---|---|---|
| Formality | Description-Only | 1.09 | 0.57 | null |
| Formality | Few-Shot | **1.55** | **0.0029** | **significant — formal favored** |
| Formality | CoT | 1.09 | 0.56 | null |
| Implied Geography | Description-Only | 0.82 | 0.071 | trend (global_south disfavored) |
| Implied Geography | Few-Shot | **0.71** | **0.0019** | **significant — global_south disfavored** |
| Implied Geography | CoT | **0.40** | **3.5e-13** | **significant, large — global_south heavily disfavored** |

## The genuinely surprising finding

**Chain-of-Thought shows the single largest bias in the entire matrix**
(implied geography, OR=0.40 — a global-south-framed client has less than
half the odds of an identical-utility global-north-framed client of being
selected), **while showing zero formality bias** (OR=1.09, indistinguishable
from the control). This cuts against the intuitive expectation that
explicit step-by-step reasoning would make a model's selections *more*
grounded in the stated numeric facts and therefore *less* swayed by
phrasing — CoT is the most biased strategy on one axis and the least
biased on the other, not uniformly better or worse than Description-Only
or Few-Shot. A plausible mechanism worth stating carefully in the
paper's discussion (as a hypothesis, not a demonstrated cause): forcing
the model to write out a rationale gives it more surface area to invoke
institutional/infrastructural associations tied to the geography axis's
wording ("community-center device," "intermittent connectivity") that the
formality axis's wording doesn't carry, since formality varies register
without invoking a socioeconomic institutional frame.

**Few-Shot is the only strategy significant on *both* axes**, and is the
one strategy whose prompt includes worked examples with explicit
loss-based selection criteria before the real pool — worth noting since
one might have expected worked examples emphasizing the numeric decision
rule to *anchor* the model on the stated criteria and reduce phrasing
sensitivity, not increase it. Both this and the CoT finding are the kind
of non-obvious, strategy-dependent results that make the "all three
strategies for rigor" scope decision back in Section 3 of the
Implementation Plan pay off — a single-strategy study would have reported
either a much weaker pooled effect (mixing CoT's null formality result
into a formal average) or missed that the two axes are driven by
different strategies entirely.

## Numeric-control sanity check (again)

Power-of-Choice's near-borderline formality result (OR=0.815, p=0.082,
*opposite* direction from every LLM strategy) is noise, not signal — it
never reads `metadata_text`, so it structurally cannot detect a phrasing
effect; a p near 0.08 out of many hypothesis tests run here is exactly the
false-positive rate one expects by chance, and its direction doesn't
support any part of the bias story. Its implied-geography result
(OR=1.013, p=0.879) is a near-perfect null, as expected. Both together
confirm the harness itself isn't the source of the LLM findings above.

## What this means for Phase 7 (Mitigation)

The strategy-specificity is itself a design input for Phase 7's
metadata-templating mitigation: since the bias isn't uniform, the
mitigation should be evaluated per (strategy, axis) cell rather than
pooled, and success should be judged by whether it collapses
**Few-Shot's formality effect** and **CoT's + Few-Shot's geography
effects** specifically toward the control's null, not just by an
average-case improvement that could mask an unresolved strategy-specific
bias.

## Provenance

Analysis run in Kaggle (CPU, no model needed — `analysis/phase6_analysis.py`
on the completed `results/phase4_pilot_clients.csv`), outputs downloaded
and committed here immediately after, before starting Phase 7, so the
finding isn't lost to memory across sessions.
