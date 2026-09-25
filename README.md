# FairAgent

**Auditing and Mitigating Semantic Selection Bias in LLM-Orchestrated Federated Learning**

## What this is

Federated Learning (FL) is designed so client-selection decisions can be made from numbers alone (loss, data volume, bandwidth) — precisely so a client's identity can't influence whether it participates. Emerging **agentic FL** systems replace this with an LLM reasoning over natural-language client metadata, explicitly to capture "nuance" numeric criteria miss. This reopens a risk numeric selection is structurally immune to: **selection influenced by how a client's metadata is *worded*, independent of the client's actual utility.* *

FairAgent designs a controlled audit to measure whether this bias exists, compares LLM-based selection against numeric baselines (Oort, Power-of-Choice) under matched-utility conditions, and evaluates a mitigation (metadata templating).

## Base paper

You, L., Liu, S., Wang, T., Zuo, B., Chang, Y., & Yuen, C. (2024). *AiFed: An Adaptive and Integrated Mechanism for Asynchronous Federated Data Mining.* IEEE Transactions on Knowledge and Data Engineering, 36(9), 4411–4427. [DOI: 10.1109/TKDE.2023.3332770](https://doi.org/10.1109/TKDE.2023.3332770)

Implemented via the [GOLF framework](https://github.com/IntelligentSystemsLab/generic_and_open_learning_federator).

Jarczewski et al.'s *Agentic Federated Learning* (arXiv:2604.04895) remains the conceptual anchor for the bias-audit question this project investigates, but AiFed — a Q1-indexed journal paper with a public runnable implementation — is the base paper for replication/extension purposes.

## Infrastructure

Fully online: GitHub (code) + Kaggle Notebooks (compute) + Kaggle Datasets (model weights, results) — no local machine required. See `configs/` for run configurations and the project's Implementation Plan for the full session-discipline pattern (every session pushes results out before ending; nothing lives only on ephemeral notebook disk).

- **Models:** Llama-3.2-3B-Instruct (Core orchestrator), served via Hugging Face `transformers` — packaged as a private Kaggle Dataset. Llama-3.1-8B-Instruct and Qwen3-8B are Stretch-scope, pending a disk-constrained download.
- **Results:** versioned in a separate Kaggle Dataset, pushed after each work session.

## Repo structure

| Folder | Contents |
|---|---|
| `fl_core/` | CNN model, Dirichlet non-IID CIFAR-10 partitioning, FedAvg training loop |
| `client_selectors/` | Pluggable selection interface + Random, Power-of-Choice, and Oort baseline selectors (numeric, bias-free-by-construction controls) |
| `metadata/` | Synthetic client metadata generator and phrasing-variant library (Phase 3) |
| `configs/` | Run configurations (YAML) |
| `experiments/` | Trial-runner scripts |
| `analysis/` | Plots, statistical analysis notebooks, and result notes |
| `results/` | Pointer to the canonical Kaggle results Dataset (raw trial data lives there, not in-repo) |
| `paper/` | Manuscript draft (`manuscript.md`) — Phase 10 |

## Status

**Phase 1 — Core FL Harness: complete (corrected).** Small CNN trained via FedAvg on Dirichlet-partitioned (α=0.5) CIFAR-10, 20 simulated clients, pluggable selector interface, three numeric baselines implemented and sanity-checked over 30 rounds:

| Selector | Final test accuracy |
|---|---|
| Random | 62.05% |
| Power-of-Choice | 63.38% |
| Oort | 59.27% |

Harness converges sanely (Gate Check #1 passed). **Correction note:** an earlier bug (`update_stats()` never wired into the training loop, so PoC/Oort never received real per-round loss feedback) was found during Phase 2 integration testing and fixed; these numbers supersede all previously documented Phase 1 results. PoC now clearly beats Random as expected. Oort still underperforms Random — a real, documented characteristic of the simplified exploration formula, not a bug; see `analysis/phase1_notes.md` for full discussion.

**Phase 2 — Orchestrator Integration: complete.** LLM-based selector (Llama-3.2-3B-Instruct) implemented with all three Core prompting strategies (Description-Only, Few-Shot, CoT) via `client_selectors/llm_base.py` + strategy subclasses, smoke-tested (0 retries, no fallback).

**Phase 3 — Synthetic Metadata Design: complete.** `metadata/schema.py` (24 fixed `TrueUtility` client identities, seed=7) and `metadata/profiles.py` (Formality: formal/casual; Implied Geography: global_north/global_south institutional framing, no named countries) — matched-pair design where every phrasing variant of a client states identical numeric facts. Validity-checked (human read, judged genuine not strawman) and automated numeric-consistency checked.

**Phase 4 — Selection-Trial Infrastructure + Pilot: complete.** `experiments/run_selection_trials.py`: single-shot selection-trial runner built on the session-discipline pattern (resume from existing results, incremental flush-per-row, bounded batch, push-before-stop). Pilot (`configs/phase4_pilot.yaml`, 3 strategies x 2 axes x 40 reps = 240 LLM trials, plus a same-pool numeric-control negative control per trial) completed across several bounded Kaggle sessions. `client_selectors/llm_base.py`'s `_render_client_line` renders `ClientProfile.metadata_text` (Phase 3's phrasing variants) verbatim when populated, instead of only the Phase 2 neutral numeric line — this is what makes trials actually test the phrasing variable.

**Pilot result (see `analysis/phase4_notes.md`):** holding true utility and list position fixed, **Formality already shows a significant selection bias** — formal-register clients selected at 1.44x the odds of identical-utility casual-register clients (p=0.0098). **Implied Geography trends the same direction** (global-south framing selected less, OR=0.77) but isn't significant yet at the pilot's 40 trials/cell (p=0.060) — the pilot's own power calculation (`analysis/phase4_power_calc_results.md`) says this axis needs ~221 trials/cell, not 40.

**Phase 5 — Core Trial Matrix: complete.** 1050 trials collected (120/cell formality, 230/cell implied geography, `configs/phase5_core_matrix.yaml`), continuing Phase 4's 240 trials by `trial_id`.

**Phase 6 — Statistical Analysis: complete.** `analysis/phase6_analysis.py`: chi-square + cluster-robust logistic regression per (axis, strategy/control) cell. Full results in `analysis/phase6_results.md` / `.csv`, forest plots in `analysis/phase6_forest_*.png`, interpretation in `analysis/phase6_notes.md`.

**Headline finding:** the phrasing bias is real but strategy- and axis-specific, not uniform. On identical true-utility pools, the numeric control (Power-of-Choice) shows no phrasing effect on either axis (as it must — it never reads client metadata), while three of six LLM (strategy x axis) cells show a significant effect: **Few-Shot** is biased on *both* axes (formality OR=1.55, p=0.0029; geography OR=0.71, p=0.0019); **CoT** shows the single largest bias in the matrix on geography (OR=0.40, p=3.5e-13) but *no* formality bias (OR=1.09, p=0.56); **Description-Only** shows a geography trend (OR=0.82, p=0.071) that doesn't reach significance. Chain-of-Thought reasoning does not uniformly reduce phrasing sensitivity — it is the most biased strategy on one axis and the least on the other.

**Phase 7 — Mitigation (statistical half): complete.** `metadata/profiles.py`'s `render_neutral()` — one fixed canonical rendering of a client's true utility, same facts, no register/framing — replaces phrasing variants (`use_neutral_template: true`, `configs/phase7_mitigation.yaml`). Re-ran the exact same pools as Phase 5/6 (matched by `trial_id`) at 60 trials/cell. Results in `analysis/phase7_mitigation_results.md` / `.csv`, interpretation in `analysis/phase7_mitigation_notes.md`.

**Mitigation result:** every previously-biased cell collapsed to a non-significant, control-like odds ratio. Few-Shot's formality bias (OR=1.55, p=0.0029 -> OR=1.17, p=0.54) and geography bias (OR=0.71, p=0.0019 -> OR=1.13, p=0.61) both vanish; CoT's large geography bias (OR=0.40, p=3.5e-13 -> OR=0.96, p=0.80) vanishes; Description-Only's geography trend (OR=0.82, p=0.071 -> OR=1.00, p=1.00) vanishes. Nothing moved in the wrong direction and the numeric control stayed null throughout, confirming the collapse is the mitigation working, not noise in the harness.

**Phase 7 — Quality Trade-off Check: complete.** 4 full FedAvg runs (`configs/phase7_quality_check.yaml`, `experiments/run_phase7_quality_check.py`) comparing phrased vs. neutral metadata for Few-Shot/formality and CoT/implied_geography. Results in `analysis/phase7_quality_results.md`, curves in `analysis/phase7_quality_curves.png`, interpretation in `analysis/phase7_quality_notes.md`.

**Quality result:** neutral templating costs a small, consistent accuracy penalty — CoT/geography drops 1.12 points (0.5866 -> 0.5754), Few-Shot/formality drops 0.64 points (0.5906 -> 0.5842). Both directions consistent (never a gain), but this is a single seeded run per condition, not a multi-seed average — the gap is too small to distinguish from ordinary FedAvg run-to-run variance without repeated seeds. Reported as an honest limitation, not a validated statistical result, but the direction across both tested cells is at least suggestive that the mitigation is not entirely free.

**Phase 7 is now fully complete** — both the statistical mitigation (bias collapses to null) and the quality trade-off check (small, honestly-reported cost) are committed.

**Statistical hygiene: complete.** Multiple-comparison correction (Bonferroni + Benjamini-Hochberg) applied to Phase 6's 6 LLM-strategy hypothesis tests — all three headline findings survive both corrections; see `analysis/phase6_multiple_comparisons.md`. (Phase 2's AiFed replication checkpoint was descoped per program guidance — a base-paper citation and rationale is required, not an experimental replication; both are satisfied above and in `paper/manuscript.md`.) Phase 8/9 (Stretch: second/third orchestrator model, remaining variation axes, FEMNIST) intentionally out of scope for this submission, per the Implementation Plan's own Core-vs-Stretch framing.

**Phase 10 — Manuscript: draft in progress.** `paper/manuscript.md` — full IMRaD draft (abstract, intro, related work, methods, results, mitigation, limitations, conclusion, references) built from the analysis notes above. Needs: deeper related-work literature review, journal-template formatting, and a final read-through before submission.
