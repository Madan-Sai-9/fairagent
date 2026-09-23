"""Phase 1 sanity check: runs FedAvg once per selector (Random, Power-of-Choice,
Oort) on identically-partitioned non-IID CIFAR-10, to confirm the harness
converges sanely (Gate Check #1) before any further phases build on it."""

import sys
sys.path.insert(0, "/kaggle/working")

import copy
import yaml
import torch

from fl_core.model import SmallCNN
from fl_core.data import load_cifar10, dirichlet_partition, client_subsets
from fl_core.fedavg import run_fedavg
from client_selectors.base import ClientProfile
from client_selectors.random_selector import RandomSelector
from client_selectors.power_of_choice import PowerOfChoiceSelector
from client_selectors.oort import OortSelector


def main(config_path: str = "/kaggle/working/configs/phase1_sanity.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    train_set, test_set = load_cifar10()
    client_idx = dirichlet_partition(
        train_set, cfg["num_clients"], cfg["dirichlet_alpha"], seed=cfg["seed"]
    )
    client_datasets = client_subsets(train_set, client_idx)

    client_profiles = [
        ClientProfile(client_id=i, num_samples=len(client_idx[i]), stats={})
        for i in range(cfg["num_clients"])
    ]

    selectors = {
        "random": RandomSelector(client_profiles, seed=cfg["seed"]),
        "power_of_choice": PowerOfChoiceSelector(client_profiles, seed=cfg["seed"]),
        "oort": OortSelector(client_profiles, seed=cfg["seed"]),
    }

    histories = {}
    for name, selector in selectors.items():
        print(f"\n{'='*20} Running FedAvg with selector: {name} {'='*20}")
        model = SmallCNN()
        _, acc_history = run_fedavg(
            model=model,
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
        histories[name] = acc_history
        print(f"{name}: final test accuracy = {acc_history[-1]:.4f}")

    return histories


if __name__ == "__main__":
    main()
