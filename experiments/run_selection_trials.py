"""
experiments/run_selection_trials.py

Phase 4: Selection-Trial Infrastructure + Pilot.

Runs single-shot client-selection trials to measure whether LLM-orchestrated
selection depends on client-metadata PHRASING (Formality / Implied
Geography, metadata/profiles.py) independent of a client's TRUE utility
(metadata/schema.py). Each trial:

  1. draws `pool_size` client identities from the 24 fixed TrueUtility
     profiles (seed=7 — same identities every session, matches Phase 3),
  2. independently coin-flips each pooled client's phrasing VARIANT for the
     one axis under test, so variant is uncorrelated with true utility
     within a trial — the manipulation Phase 6's regression isolates,
  3. shuffles pool order, so list position is uncorrelated with identity/
     variant — the covariate Phase 6 explicitly controls for,
  4. calls one LLM strategy's select(k) on that pool, and, on the identical
     pool/order, a numeric baseline's select(k) as a same-pool NEGATIVE
     CONTROL (numeric selectors never read metadata_text, so any variant
     "effect" there is a confound check, not signal),
  5. logs one trial-level row and one per-(client, selector) row.

Session-discipline pattern (Implementation Plan Section 2):
  - on start, loads existing result files and skips already-completed
    trial_ids — safe to stop and resume across sessions/devices.
  - flushes each row to disk immediately, nothing held for "the end."
  - runs a bounded batch: stops at `batch_size` new trials OR
    `max_minutes_per_session` wall-clock, whichever comes first.
  - the caller (the Kaggle notebook) pushes the two output CSVs out as a
    new results/ Kaggle Dataset version, and any code changes back to
    GitHub, once this returns — that push is not this script's job.

Usage (from a Kaggle notebook, after loading the HF tokenizer/model):
    from experiments.run_selection_trials import run_pilot
    run_pilot(tokenizer, model, device)
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import time

import yaml

from client_selectors.base import ClientProfile
from client_selectors.registry import build_selector, register_llm_selectors
from metadata.profiles import AXES, render
from metadata.schema import generate_true_utilities

TRIAL_FIELDS = [
    "trial_id", "strategy", "axis", "pool_size", "k",
    "retry_count", "fallback_triggered", "rationale", "raw_output_truncated",
    "timestamp",
]

CLIENT_FIELDS = [
    "trial_id", "selector", "axis", "client_id", "variant", "list_position",
    "dataset_size", "historical_accuracy", "historical_loss", "bandwidth_mbps",
    "rounds_participated", "selected",
]


def _trial_seed(trial_id: str) -> int:
    """Deterministic seed derived from the trial_id string, so re-running
    (or resuming) the same trial_id always reproduces the same pool/variant
    assignment/order regardless of run order or session."""
    return int(hashlib.sha256(trial_id.encode()).hexdigest()[:8], 16)


def _load_done_trial_ids(trial_results_path: str) -> set:
    if not os.path.exists(trial_results_path):
        return set()
    with open(trial_results_path, newline="") as f:
        return {row["trial_id"] for row in csv.DictReader(f)}


def _enumerate_trial_plan(cfg) -> list:
    """Deterministic full list of (strategy, axis, trial_id) for the pilot's
    cells. Order here doesn't affect resume-correctness (only trial_id
    membership in the results file does) but is kept stable for readability."""
    plan = []
    for strategy in cfg["strategies"]:
        for axis in cfg["axes"]:
            for rep in range(cfg["trials_per_cell"]):
                plan.append((strategy, axis, f"{strategy}__{axis}__{rep:04d}"))
    return plan


def _build_pool(true_utilities, axis, pool_size, trial_id):
    """Draws `pool_size` client identities, assigns each an independent
    random variant for `axis`, shuffles order. Returns a list of
    (client_id, true_utility, variant) tuples in trial (shuffled) order."""
    rng = random.Random(_trial_seed(trial_id))
    variant_names = list(AXES[axis].keys())
    all_ids = list(range(len(true_utilities)))
    pool_ids = rng.sample(all_ids, pool_size)
    entries = [(cid, true_utilities[cid], rng.choice(variant_names)) for cid in pool_ids]
    rng.shuffle(entries)
    return entries


def _make_llm_profiles(entries, axis):
    return [
        ClientProfile(
            client_id=cid, num_samples=u.dataset_size, stats={},
            metadata_text=render(axis, variant, cid, u),
        )
        for cid, u, variant in entries
    ]


def _make_control_profiles(entries):
    return [
        ClientProfile(client_id=cid, num_samples=u.dataset_size, stats={})
        for cid, u, _ in entries
    ]


def _append_row(path, fieldnames, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _log_client_rows(client_csv_path, trial_id, selector_name, axis, entries, selected_ids):
    selected_set = set(selected_ids)
    for pos, (cid, u, variant) in enumerate(entries):
        _append_row(client_csv_path, CLIENT_FIELDS, {
            "trial_id": trial_id,
            "selector": selector_name,
            "axis": axis,
            "client_id": cid,
            "variant": variant,
            "list_position": pos,
            "dataset_size": u.dataset_size,
            "historical_accuracy": u.historical_accuracy,
            "historical_loss": u.historical_loss,
            "bandwidth_mbps": u.bandwidth_mbps,
            "rounds_participated": u.rounds_participated,
            "selected": int(cid in selected_set),
        })


def run_pilot(tokenizer, model, device, config_path: str = "configs/phase4_pilot.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    register_llm_selectors(tokenizer, model, device)

    true_utilities = generate_true_utilities(
        num_clients=cfg["num_true_clients"], seed=cfg["metadata_seed"]
    )

    os.makedirs(cfg["results_dir"], exist_ok=True)
    trial_csv = os.path.join(cfg["results_dir"], cfg["trial_results_file"])
    client_csv = os.path.join(cfg["results_dir"], cfg["client_results_file"])

    done = _load_done_trial_ids(trial_csv)
    plan = _enumerate_trial_plan(cfg)
    todo = [t for t in plan if t[2] not in done]

    print(f"Phase 4 pilot: {len(plan)} planned trials, {len(done)} already done, "
          f"{len(todo)} remaining.")

    start = time.time()
    max_seconds = cfg["max_minutes_per_session"] * 60
    completed_this_session = 0

    for strategy, axis, trial_id in todo:
        if completed_this_session >= cfg["batch_size"]:
            print(f"Batch size {cfg['batch_size']} reached — stopping.")
            break
        if time.time() - start >= max_seconds:
            print(f"Wall-clock budget ({cfg['max_minutes_per_session']} min) reached — stopping.")
            break

        entries = _build_pool(true_utilities, axis, cfg["pool_size"], trial_id)
        k = cfg["clients_per_trial"]

        llm_profiles = _make_llm_profiles(entries, axis)
        llm_selector = build_selector(strategy, client_profiles=llm_profiles, seed=_trial_seed(trial_id))
        llm_selected = llm_selector.select(k)

        _log_client_rows(client_csv, trial_id, strategy, axis, entries, llm_selected)
        _append_row(trial_csv, TRIAL_FIELDS, {
            "trial_id": trial_id,
            "strategy": strategy,
            "axis": axis,
            "pool_size": cfg["pool_size"],
            "k": k,
            "retry_count": llm_selector.last_retry_count,
            "fallback_triggered": int(llm_selector.last_retry_count >= llm_selector.max_retries),
            "rationale": (llm_selector.last_rationale or "")[:500],
            "raw_output_truncated": (llm_selector.last_raw_output or "")[:500],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        control_name = cfg["control_selector"]
        control_profiles = _make_control_profiles(entries)
        control_selector = build_selector(control_name, client_profiles=control_profiles, seed=_trial_seed(trial_id))
        for cid, u, _ in entries:
            control_selector.update_stats(cid, loss=u.historical_loss)
        control_selected = control_selector.select(k)
        _log_client_rows(client_csv, trial_id, control_name, axis, entries, control_selected)

        completed_this_session += 1
        if completed_this_session % 10 == 0:
            print(f"  {completed_this_session} trials this session "
                  f"({len(done) + completed_this_session}/{len(plan)} total)...")

    print(f"Session done: {completed_this_session} new trials "
          f"({len(done) + completed_this_session}/{len(plan)} total planned). "
          f"Remember to push {trial_csv} and {client_csv} out as a new results/ "
          f"Kaggle Dataset version, and commit any code changes, before logging off.")
    return trial_csv, client_csv


if __name__ == "__main__":
    raise SystemExit(
        "run_selection_trials.run_pilot() needs a loaded HF tokenizer/model/device "
        "(GPU work) — call it from the Kaggle notebook after loading Llama-3.2-3B-"
        "Instruct, e.g.:\n\n"
        "    from experiments.run_selection_trials import run_pilot\n"
        "    run_pilot(tokenizer, model, device)\n"
    )
