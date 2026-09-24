# Phase 7 — Mitigation (Neutral Templating): Results & Notes

**Trial design:** identical pools to Phase 5/6 (same `trial_id` -> same
client pool, variant assignment, and list order — verified locally before
running), 60 trials/cell instead of 120/230, every client rendered
through `metadata/profiles.py`'s `render_neutral()` (one fixed template
covering the same 5 TrueUtility fields) instead of its assigned
formality/geography phrasing.

## Result: mitigation succeeded, cleanly, across every previously-biased cell

| Axis | Strategy | Phase 6 (biased) OR, p | Phase 7 (mitigated) OR, p | Verdict |
|---|---|---|---|---|
| Formality | Few-Shot | 1.55, **0.0029** | 1.17, 0.54 | **collapsed to null** |
| Implied Geography | Few-Shot | 0.71, **0.0019** | 1.13, 0.61 | **collapsed to null** |
| Implied Geography | CoT | 0.40, **3.5e-13** | 0.96, 0.80 | **collapsed to null** |
| Implied Geography | Description-Only | 0.82, 0.071 (trend) | 1.00, 1.00 | trend gone |
| Formality | CoT | 1.09, 0.56 (already null) | 1.04, 0.85 | stays null |
| Formality | Description-Only | 1.09, 0.57 (already null) | 1.37, 0.18 | stays non-significant |
| Formality | Power-of-Choice (control) | 0.815, 0.082 | 0.79, 0.16 | null, as always |
| Implied Geography | Power-of-Choice (control) | 1.01, 0.88 | 0.88, 0.42 | null, as always |

Every cell that showed a significant or trending bias in Phase 6 —
Few-Shot on both axes, CoT's large geography effect, Description-Only's
geography trend — lands within a non-significant, control-like odds
ratio range (0.79-1.37) once metadata is rendered neutrally. Nothing
moved in the "wrong" direction (no cell became newly significant), and
the numeric control stays exactly where it should on both re-runs
(near-null, unaffected by the manipulation either way, since it never
read `metadata_text` before or after).

## Why this is a strong result, not just an expected one

Collapsing to null was the intended outcome, but it wasn't guaranteed
mechanically — the template still had to actually work (correctly convey
every numeric fact, in a form the LLM could still use for genuine
utility-based selection) rather than just deleting information the model
happened to rely on for good decisions. That the effect vanishes cleanly,
symmetrically, across three structurally different prompting strategies
(Description-Only, Few-Shot, CoT) and both axes, without introducing any
new spurious effect, is the evidence the mitigation is doing what it
claims: removing the phrasing signal while preserving the utility signal.
Whether it preserves utility *well enough* — i.e. whether selection
quality (not just selection fairness) holds up — is a separate question,
answered by the quality trade-off check (full FL runs), not by this
statistical re-run alone.

## Honest caveats

- **Reduced power, by design.** 60 trials/cell is enough to distinguish
  "collapsed to ~1.0" from Phase 6's large effects (OR 0.40-1.55), but
  a small residual bias (say OR 1.1-1.2) could exist without reaching
  significance here. The mitigation should be read as "no longer
  producing the large, previously-detected biases," not "provably zero
  bias at any effect size."
- **Description-Only's formality OR moved from 1.09 to 1.37** — still
  non-significant (p=0.18) and within a plausible range for a selector
  that was already null, but worth noting as the one cell that drifted
  rather than staying flat; likely sampling noise given n=60, not a
  reversal of the mitigation's effect (there was no formality bias to
  mitigate in this cell to begin with).

## What's still needed for Phase 7 to be complete

The plan's second deliverable — **a quality trade-off check via a
handful of full-FL runs**, comparing final model accuracy under the
original phrased metadata vs. neutral templating for the strategies that
showed bias — hasn't been run yet. A mitigation that removes bias but
degrades the model's ability to select genuinely high-utility clients
would be a real cost worth reporting, not a free win. This is the
natural next step before Phase 7's push checklist is fully satisfied.

## Provenance

Mitigation trials run in Kaggle (GPU, Llama-3.2-3B-Instruct) across
several bounded sessions; analysis run via `analysis/phase6_analysis.py`
reused with `output_prefix="phase7_mitigation"`. Outputs downloaded and
committed here immediately after.
