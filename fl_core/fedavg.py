"""Plain-Python FedAvg training loop. No RPC/simulation framework needed since
everything runs in a single process — kept swappable for a real framework later
if the project ever needs true distributed execution."""

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
    Returns the resulting state_dict and the number of samples trained on
    (needed for FedAvg's weighted averaging)."""
    local_model = copy.deepcopy(model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(x), y)
            loss.backward()
            optimizer.step()

    return get_flat_state(local_model), len(dataset)


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
    """Runs `num_rounds` of FedAvg, using `selector` to pick `clients_per_round`
    clients each round. Returns the trained global model and a per-round
    test-accuracy history (for plotting / gate-check verification)."""
    model = model.to(device)
    accuracy_history = []

    for round_num in range(1, num_rounds + 1):
        selected_ids = selector.select(clients_per_round)

        states_and_counts = []
        for client_id in selected_ids:
            state, n_samples = local_train(
                model, client_datasets[client_id], local_epochs, lr, batch_size, device
            )
            states_and_counts.append((state, n_samples))

        new_global_state = fedavg_aggregate(states_and_counts)
        set_flat_state(model, new_global_state)

        acc = evaluate(model, test_dataset, device)
        accuracy_history.append(acc)
        print(f"Round {round_num:3d}/{num_rounds} | selected {selected_ids} | test acc: {acc:.4f}")

    return model, accuracy_history
