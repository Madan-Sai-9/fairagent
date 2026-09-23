"""CIFAR-10 loading and Dirichlet non-IID partitioning across simulated clients."""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms


def load_cifar10(data_dir: str = "/kaggle/working/data"):
    """Downloads (if needed) and returns the CIFAR-10 train/test torchvision datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_set, test_set


def dirichlet_partition(dataset: Dataset, num_clients: int, alpha: float, seed: int = 42):
    """Partitions `dataset` indices across `num_clients` using a Dirichlet(alpha)
    distribution per class — the standard non-IID FL partitioning method.

    Lower alpha -> more label-skewed (non-IID) clients.
    Higher alpha -> closer to IID (uniform) split.

    Returns: list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        # Dirichlet proportions of this class's samples across clients
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # convert proportions to cumulative split points
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)

        for client_id, idx in enumerate(splits):
            client_indices[client_id].extend(idx.tolist())

    # shuffle each client's own index list
    for idx_list in client_indices:
        rng.shuffle(idx_list)

    return client_indices


def client_subsets(dataset: Dataset, client_indices):
    """Wraps each client's index list as a torch Subset."""
    return [Subset(dataset, idx) for idx in client_indices]
