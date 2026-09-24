# FairAgent

**Auditing and Mitigating Semantic Selection Bias in LLM-Orchestrated Federated Learning**

## What this is

Federated Learning (FL) is designed so client-selection decisions can be made from numbers alone (loss, data volume, bandwidth) — precisely so a client's identity can't influence whether it participates. Emerging **agentic FL** systems replace this with an LLM reasoning over natural-language client metadata, explicitly to capture "nuance" numeric criteria miss. This reopens a risk numeric selection is structurally immune to: **selection influenced by how a client's metadata is *worded*, independent of the client's actual utility.**

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

**Phase 4 — Selection-Trial Infrastructure + Pilot (in progress).** `experiments/run_selection_trials.py`: single-shot selection-trial runner built on the session-discipline pattern (resume from existing results, incremental flush-per-row, bounded batch, push-before-stop). `configs/phase4_pilot.yaml` targets 240 LLM trials (3 strategies x 2 axes x 40 reps) plus a same-pool numeric-control (Power-of-Choice) negative control per trial. `client_selectors/llm_base.py`'s `_render_client_line` now renders `ClientProfile.metadata_text` (Phase 3's phrasing variants) verbatim when populated, instead of only the Phase 2 neutral numeric line — this is what makes trials actually test the phrasing variable. `analysis/phase4_power_calc.py` fits the pilot's observed variant effect (clustered logistic regression) to set Phase 5's Core trial-count target.

**Next:** run the Phase 4 pilot batches in Kaggle (GPU session, Llama-3.2-3B-Instruct), then the power calculation to lock Phase 5's Core trial-count target.
