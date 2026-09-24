# Phase 4 — Selection-Trial Pilot: Results & Notes

**Config:** `configs/phase4_pilot.yaml` — Llama-3.2-3B-Instruct, 3 prompting
strategies x 2 axes x 40 reps = 240 LLM trials, pool_size=10, k=3,
Power-of-Choice as the same-pool numeric-control negative control.
240/240 pilot trials completed across multiple bounded Kaggle sessions
(session-discipline pattern — resumed cleanly by `trial_id` each time).

## Pilot power-calc results (see `analysis/phase4_power_calc_results.md`)

Clustered logistic regression (`selected ~ variant + true-utility
covariates + list_position`, clustered by `trial_id`), LLM-strategy rows
only (1200 client-selector rows per axis; numeric-control rows excluded
by construction — see below):

| Axis | Reference -> treatment | Odds ratio | p-value | Phase 5 target (trials/cell) |
|---|---|---|---|---|
| Formality | casual -> formal | 1.4367 | **0.0098** | ~113 |
| Implied geography | global_north -> global_south | 0.7693 | 0.0598 | ~221 |

## Honest read of this result

**This is the central finding the project is designed to detect, and the
pilot already shows it in one axis at conventional significance.**
Holding true utility (dataset size, historical accuracy/loss, bandwidth,
rounds participated) and list position fixed, a client described in
**formal** register is selected ~44% more often (in odds) than the
identical-utility client described **casually** (p=0.0098). Two clients
with numerically identical facts get different LLM-orchestrator selection
outcomes purely because of how those facts are phrased — exactly the
bias this project set out to audit for.

**Implied geography trends the same direction as expected** (global-south
institutional framing selected less often than global-north framing,
OR=0.77) but the pilot's 240 trials don't clear p<0.05 yet (p=0.060).
This is not a null result — it's an underpowered one; the pilot's own
power calculation says so directly (needs ~221 trials/cell vs. the 40/cell
collected so far). Phase 5 is sized precisely to resolve this.

**Numeric-control sanity check:** Power-of-Choice was run on every trial's
identical client pool as a same-pool negative control. It never reads
`metadata_text`, so its selections cannot depend on phrasing variant by
construction — this is a construct-validity check on the harness itself
(confirms the LLM effect above isn't an artifact of the pool-construction
code), not part of the power calculation. Full quantitative comparison
(agentic vs. numeric disparity) is Phase 6's job, run on the completed
Core matrix, not repeated per-pilot-batch.

**Methodological note for Phase 5:** the trial-count targets above were
fixed once, from this one pilot power calculation — Phase 5 collects to
that pre-set target and analyzes once at the end (Phase 6), rather than
re-running the power calc or a significance test after each batch and
stopping early once p<0.05. Re-testing on every batch would inflate the
false-positive rate; the pilot's job was only to size Phase 5's N, not to
be the first of a series of interim looks.

## Provenance

Pilot run across several bounded Kaggle GPU sessions (session-discipline
pattern: resume-by-`trial_id`, flush-per-row, bounded batch, push before
logging off) using `notebooks/phase4_pilot.ipynb`. Full 240/240 trial
completion and power-calc output confirmed in-session; this note written
immediately after, before starting Phase 5, so the finding isn't lost to
memory across sessions.
