"""
experiments/run_phase7_quality_check.py

Phase 7 (second deliverable): quality trade-off check.

analysis/phase7_mitigation_notes.md showed neutral templating collapses
the SELECTION bias to null. That says nothing yet about whether it costs
anything in actual FL performance — a mitigation that fixes fairness by
throwing away useful signal would be a real, reportable cost, not a free
win. This script runs full, multi-round FedAvg (same harness as Phase 1)
under two conditions — original phrased metadata vs. Phase 7's neutral
template — for the (strategy, axis) cells that showed the strongest bias
in Phase 6: Few-Shot/formality and CoT/implied_geography (CoT's geography
effect was the single largest in the whole study).

Client metadata_text is fixed for the whole run (assigned once at setup,
not regenerated each round from live per-round loss) — this deliberately
matches every trial in Phase 4-7, which tested static, institutional-
style client descriptions, not a live telemetry feed. Numeric-selector
controls still receive real per-round loss via update_stats() as always;
the LLM selectors' prompts do not, by the same design choice already made
in client_selectors/llm_base.py's _render_client_line (no numeric line is
appended alongside metadata_text, so live loss wouldn't leak into either
condition).

Deliberately "a handful of runs" (the Implementation Plan's own phrasing
for this phase) — 4 full FedAvg runs, each identical in cost to Phase 1's
runs, on the same 20-client Dirichlet-partitioned CIFAR-10 used
throughout the project. Meant for a single focused Kaggle GPU session.
Skips any run_name already present in the results file, so a session that
dies partway through the 4 runs can safely resume.

Usage (from a Kaggle notebook, after loading the HF tokenizer/model):
    from experiments.run_phase7_quality_check import run_quality_check
    run_quality_check(tokenizer, model, device)
"""

from __future__ import annotations

import csv
import os
import random

import torch
import yaml

from client_selectors.base import ClientProfile
from client_selectors.registry import build_selector, register_llm_selectors
from fl_core.data import load_cifar10, dirichlet_partition, client_subsets
from fl_core.fedavg import run_fedavg
from fl_core.model import SmallCNN
from metadata.profiles import AXES, render, render_neutral
from metadata.schema import generate_true_utilities

RESULT_FIELDS = ["run_name", "strategy", "axis", "condition", "round", "test_accuracy"]


def _assign_variants(num_clients, axis, seed):
    """One fixed variant per client for the whole run, chosen independently
    of that client's real training data — mirrors the trial design's
    variant-uncorrelated-with-utility manipulation."""
    rng = random.Random(seed)
    variant_names = list(AXES[axis].keys())
    return {cid: rng.choice(variant_names) for cid in range(num_clients)}


def _build_client_profiles(true_utilities, client_idx, axis, variants, use_neutral_template):
    profiles = []
    for cid, u in enumerate(true_utilities):
        text = render_neutral(cid, u) if use_neutral_template else render(axis, variants[cid], cid, u)
        profiles.append(ClientProfile(
            client_id=cid, num_samples=len(client_idx[cid]), stats={}, metadata_text=text,
        ))
    return profiles


def _load_done_run_names(results_path) -> set:
    if not os.path.exists(results_path):
        return set()
    with open(results_path, newline="") as f:
        return {row["run_name"] for row in csv.DictReader(f)}


def _append_rows(path, rows):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_quality_check(tokenizer, model, device, config_path: str = "configs/phase7_quality_check.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    register_llm_selectors(tokenizer, model, device)

    torch.manual_seed(cfg["seed"])
    train_set, test_set = load_cifar10()
    client_idx = dirichlet_partition(train_set, cfg["num_clients"], cfg["dirichlet_alpha"], seed=cfg["seed"])
    client_datasets = client_subsets(train_set, client_idx)

    true_utilities = generate_true_utilities(num_clients=cfg["num_clients"], seed=cfg["metadata_seed"])

    os.makedirs(cfg["results_dir"], exist_ok=True)
    results_path = os.path.join(cfg["results_dir"], cfg["results_file"])
    done_runs = _load_done_run_names(results_path)

    for run_cfg in cfg["runs"]:
        name = run_cfg["name"]
        if name in done_runs:
            print(f"skip (already done): {name}")
            continue

        axis = run_cfg["axis"]
        variants = _assign_variants(cfg["num_clients"], axis, seed=cfg["seed"])
        profiles = _build_client_profiles(
            true_utilities, client_idx, axis, variants,
            use_neutral_template=run_cfg["use_neutral_template"],
        )

        selector = build_selector(run_cfg["strategy"], client_profiles=profiles, seed=cfg["seed"])

        print(f"=== Running: {name} ===")
        model_instance = SmallCNN()
        _, acc_history = run_fedavg(
            model=model_instance,
            client_datasets=client_datasets,
            test_dataset=test_set,
            selector=selector,
            num_rounds=cfg["num_rounds"],
            clients_per_round=cfg["clients_per_round"],
            local_epochs=cfg["local_epochs"],
            lr=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            device=device,
        )

        rows = [
            {
                "run_name": name,
                "strategy": run_cfg["strategy"],
                "axis": axis,
                "condition": "neutral" if run_cfg["use_neutral_template"] else "phrased",
                "round": r + 1,
                "test_accuracy": acc,
            }
            for r, acc in enumerate(acc_history)
        ]
        _append_rows(results_path, rows)
        print(f"{name}: final test accuracy = {acc_history[-1]:.4f}")

    print("Quality check session done. Push", results_path, "out as usual.")
    return results_path


if __name__ == "__main__":
    raise SystemExit(
        "run_quality_check() needs a loaded HF tokenizer/model/device (GPU work) — "
        "call it from the Kaggle notebook after loading Llama-3.2-3B-Instruct, e.g.:\n\n"
        "    from experiments.run_phase7_quality_check import run_quality_check\n"
        "    run_quality_check(tokenizer, model, device)\n"
    )
