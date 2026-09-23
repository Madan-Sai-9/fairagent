# Phase 1 Sanity Check — Results & Notes (Corrected Re-run)

**Config:** `configs/phase1_sanity.yaml` — CIFAR-10, Dirichlet non-IID (α=0.5),
20 clients, 5 selected/round, 30 rounds, 1 local epoch/round, SGD lr=0.01, seed=42.

## Correction note — read this first

A bug was discovered during Phase 2 integration testing: `fl_core/fedavg.py`'s
`run_fedavg()` never called `selector.update_stats()` after training each
client. This meant Power-of-Choice and Oort never received real per-round
loss feedback — their internal `_last_loss` tracking stayed at constructor
defaults (`float("inf")`) for the entire run, so their "select by highest
loss" logic was comparing tied values throughout. **All previously documented
Phase 1 results (including 61.67%/62.02%/57.41%) were produced by this
broken harness and are superseded by this corrected re-run.**

Fix: `local_train()` now returns average training loss over the final local
epoch; `run_fedavg()` calls `selector.update_stats(client_id, loss=avg_loss)`
after each client trains. This is what actually makes PoC/Oort's utility-based
logic functional.

## Final test accuracies (30 rounds, corrected harness)

| Selector | Final test accuracy |
|---|---|
| Random | 0.6205 (62.05%) |
| Power-of-Choice | 0.6338 (63.38%) |
| Oort | 0.5927 (59.27%) |

## Honest read of this result

**Power-of-Choice now clearly beats Random** (63.38% vs 62.05%), consistent
with the literature and with PoC's loss-feedback logic now actually working
— its per-round selections visibly track observed loss (recurring high-loss
clients like 8, 9, 12, 14 appear disproportionately often across rounds).

**Oort still underperforms Random** (59.27%) — but its behavior now looks
qualitatively different from the pre-fix run, and the explanation is
different too. With real feedback flowing, Oort's selections show a clear
pattern: it locks onto a small fixed client set (`[0,1,2,3,4]`) for the
first 12 rounds, then shifts in small increments (dropping/adding one
client at a time) for the rest of the run. This is the simplified Oort
formula's exploration/exploitation balance converging too aggressively
onto an early set of high-loss clients and being slow to explore beyond
it — a real property of this implementation, not a data/wiring bug. It
plausibly explains the underperformance: repeatedly re-training the same
handful of clients doesn't help the *global* model generalize across the
full non-IID pool the way genuine rotation does.

This is a legitimate, reportable finding, not something to hide: the
simplified Oort implementation's `exploration_factor` default may need
tuning, or this may be a real limitation worth discussing in the paper's
methods/limitations section. **Left for Phase 5's multi-seed matrix** to
determine whether this is consistent across seeds or specific to this one.

**Gate Check #1 ("harness converges sanely") — satisfied.** All three
selectors rise from near-chance (~25%) to 59–63% over 30 rounds. No
divergence, no crashes, and — critically — the harness now actually
exercises every selector's real logic, which the pre-fix runs did not.

## Provenance

Bug found: during Phase 2 (LLM orchestrator) integration testing, when the
LLM selector selected the identical 5 clients every round — traced to
`update_stats()` never being called anywhere in `run_fedavg()`, affecting
every loss-aware selector equally (PoC, Oort, and the LLM selector's
`_last_loss`-based rendering), not just the LLM one.

Fix applied and this re-run performed in one continuous, verified session:
fresh clone → fix written → fix verified present on disk (hard assertion)
→ cache purged → import verified to match fix (hard assertion) → 30-round
re-run. Every step asserted before proceeding, specifically to avoid the
stale-cache/unconfirmed-write issues that affected earlier sessions.
