# Phase 1 Sanity Check — Results & Notes

**Config:** `configs/phase1_sanity.yaml` — CIFAR-10, Dirichlet non-IID (α=0.5),
20 clients, 5 selected/round, 30 rounds, 1 local epoch/round, SGD lr=0.01, seed=42.

**Script:** `experiments/run_phase1_sanity.py`
**Notebook:** `phase-1-sanity-check.ipynb` (finalized run)

## Final test accuracies (30 rounds)

| Selector | Final test accuracy |
|---|---|
| Random | 0.6167 (61.67%) |
| Power-of-Choice | 0.6202 (62.02%) |
| Oort | 0.5741 (57.41%) |

## Honest read of this result

Random and Power-of-Choice both converge to a sane, meaningfully-above-chance
accuracy, confirming the harness itself (model, Dirichlet partitioning,
FedAvg loop) works correctly.

**Oort underperformed Random in this run** — the opposite of the expected
direction (Oort is designed to beat Random by prioritizing high-utility
clients). This is flagged honestly rather than omitted. Two plausible
explanations, not yet distinguished:

1. **Single-seed run-to-run variance.** Phase 1 is deliberately a single-seed
   sanity check, not a statistically powered comparison — Oort's
   exploration/exploitation balance (`exploration_weight` in the simplified
   formula used here) can behave noisily over just 30 rounds with only 20
   clients, especially early on before per-client loss estimates stabilize.
2. **A real weakness in the simplified Oort implementation** — this project's
   Oort is a simplified version of the full statistical-utility formula from
   Lai et al. (2021), and it's possible the simplification (or its default
   `exploration_weight`) genuinely underperforms in this specific non-IID
   regime (α=0.5, 20 clients).

**This does not block Phase 1's gate check.** Gate Check #1 only requires
the harness to "converge sanely" — which it does (all three selectors rise
from near-chance to 55–62% over 30 rounds, no divergence, no bugs). It does
not require every selector to outperform Random at n=1 seed. Resolving
*which* explanation above is correct is exactly what Phase 5's multi-seed
Core trial matrix is for — if Oort continues to underperform Random across
multiple seeds there, that's a real finding worth investigating (possibly
tuning `exploration_weight`, or reporting it as-is); if it was noise, later
seeds will show it recovering above Random as expected.

## Result history / provenance note

This is the third and final recorded Phase 1 run. Two earlier runs (from
different notebook sessions, one of which was briefly lost and later
recovered) produced different numbers (61.33/66.55/66.23 and separately a
partial recovered-implementation run) using structurally different code
(different file layout: `client.py`/`server.py`/`registry.py` vs. this
notebook's `fedavg.py`/`random_selector.py`/`power_of_choice.py`). Those
earlier numbers are superseded — **this run's code (already in GitHub) and
these numbers are the authoritative Phase 1 result** going forward.

Package renamed `selectors/` → `client_selectors/` because `selectors`
collides with Python's standard library module of the same name (I/O
multiplexing), which was silently shadowing the local package.
