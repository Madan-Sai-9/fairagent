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

**Phase 1 — Core FL Harness: complete.** Small CNN trained via FedAvg on Dirichlet-partitioned (α=0.5) CIFAR-10, 20 simulated clients, pluggable selector interface, three numeric baselines implemented and sanity-checked over 30 rounds:

| Selector | Final test accuracy |
|---|---|
| Random | 61.67% |
| Power-of-Choice | 62.02% |
| Oort | 57.41% |

Harness converges sanely (Gate Check #1 passed). Note: Oort underperformed Random in this single-seed run — flagged honestly rather than hidden; see `analysis/phase1_notes.md` for the full discussion and why this doesn't block Phase 1, pending resolution in Phase 5's multi-seed trial matrix.

**Next:** Phase 2 — Orchestrator integration (Llama-3.2-3B-Instruct as a fourth, LLM-based selector) and replication checkpoint against AiFed's reported results.
