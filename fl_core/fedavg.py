"""Plain-Python FedAvg training loop. No RPC/simulation framework needed since
everything runs in a single process — kept swappable for a real framework later
if the project ever needs true distributed execution.

CORRECTED (Phase 2 fix): local_train() now returns average training loss over
the final local epoch, and run_fedavg() feeds it back to the selector via
update_stats() after each client trains. This was previously missing entirely
— PoC and Oort never received real per-round loss signal, so their "select
by highest loss" logic was comparing tied constructor-default values
(float('inf')) for the whole run. This fix is what makes their utility-based
selection logic actually functional.
"""

import copy
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_flat_state(model: nn.Module):
    return copy.deepcopy(model.state_dict())


def set_flat_state(model: nn.Module, state):
    model.load_state_dict(state)


def local_train(model: nn.Module, dataset: Dataset, epochs: int, lr: float,
                 batch_size: int, device: str):
    """Trains a copy of `model` on one client's local data for `epochs` epochs.
    Returns the resulting state_dict, the number of samples trained on, and
    the average training loss over the final epoch (needed by loss-aware
    selectors — PoC, Oort, LLM orchestrators — via update_stats())."""
    local_model = copy.deepcopy(model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_epoch_losses = []
    for epoch in range(epochs):
        final_epoch_losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(x), y)
            loss.backward()
            optimizer.step()
            final_epoch_losses.append(loss.item())

    avg_loss = sum(final_epoch_losses) / len(final_epoch_losses) if final_epoch_losses else float("inf")
    return get_flat_state(local_model), len(dataset), avg_loss


def fedavg_aggregate(states_and_counts):
    """Weighted average of client state_dicts, weighted by each client's
    local sample count — standard FedAvg aggregation."""
    total_samples = sum(n for _, n in states_and_counts)
    avg_state = copy.deepcopy(states_and_counts[0][0])
    for key in avg_state.keys():
        avg_state[key] = sum(
            state[key].float() * (n / total_samples)
            for state, n in states_and_counts
        ).to(avg_state[key].dtype)
    return avg_state


def evaluate(model: nn.Module, dataset: Dataset, device: str, batch_size: int = 256):
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_fedavg(model: nn.Module, client_datasets: List[Dataset], test_dataset: Dataset,
               selector, num_rounds: int, clients_per_round: int,
               local_epochs: int, lr: float, batch_size: int, device: str):
    """Runs `num_rounds` of FedAvg. After each client trains, calls
    selector.update_stats() with the observed local loss — required for
    loss-aware selectors (PoC, Oort, LLM orchestrators) to see fresh
    per-round signal rather than only their constructor-time defaults."""
    model = model.to(device)
    accuracy_history = []

    for round_num in range(1, num_rounds + 1):
        selected_ids = selector.select(clients_per_round)

        states_and_counts = []
        for client_id in selected_ids:
            state, n_samples, avg_loss = local_train(
                model, client_datasets[client_id], local_epochs, lr, batch_size, device
            )
            states_and_counts.append((state, n_samples))
            selector.update_stats(client_id, loss=avg_loss)

        new_global_state = fedavg_aggregate(states_and_counts)
        set_flat_state(model, new_global_state)

        acc = evaluate(model, test_dataset, device)
        accuracy_history.append(acc)
        print(f"Round {round_num:3d}/{num_rounds} | selected {selected_ids} | test acc: {acc:.4f}")

    return model, accuracy_history
