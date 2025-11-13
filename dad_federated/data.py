from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset


@dataclass
class ClientData:
    client_id: int
    train_loader: DataLoader
    val_loader: DataLoader
    num_train_samples: int
    num_val_samples: int


class ImageFolderWithIndex(datasets.ImageFolder):
    # exposes targets like standard datasets with indices
    @property
    def targets(self) -> List[int]:
        return [s[1] for s in self.samples]


class FakeImageDataset(Dataset):
    """Synthetic ImageFolder-like dataset for smoke tests.

    Exposes attributes used by ImageFolder: samples, targets, classes.
    Allows cloning with a different transform but identical labels/indices.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int = 2,
        image_size: int = 224,
        transform: transforms.Compose | None = None,
        labels: List[int] | None = None,
        seed: int = 123,
    ):
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.transform = transform

        rng = np.random.default_rng(seed)
        if labels is None:
            # Generate deterministic labels
            labels = rng.integers(low=0, high=num_classes, size=self.num_samples).tolist()
        self._labels: List[int] = [int(x) for x in labels]

        # Mimic ImageFolder's samples as (path_like, label). Path is dummy here
        self.samples: List[Tuple[str, int]] = [(str(i), self._labels[i]) for i in range(self.num_samples)]
        self._classes: List[str] = [f"class-{i}" for i in range(self.num_classes)]

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def targets(self) -> List[int]:
        return self._labels

    def clone_with_transform(self, transform: transforms.Compose) -> "FakeImageDataset":
        return FakeImageDataset(
            num_samples=self.num_samples,
            num_classes=self.num_classes,
            image_size=self.image_size,
            transform=transform,
            labels=self._labels,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Create a random RGB image as PIL Image so torchvision transforms work
        img_np = (np.random.rand(self.image_size, self.image_size, 3) * 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = self._labels[idx]
        return img, label


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


def load_imagefolder_dataset(root: str | Path, image_size: int, use_fake_if_missing: bool = True) -> Tuple[Dataset, Dataset, int]:
    root = Path(root)
    train_tfms, val_tfms = build_transforms(image_size)

    if not root.exists() or not any(root.glob("**/*")):
        if not use_fake_if_missing:
            raise FileNotFoundError(f"Dataset root '{root}' does not exist or is empty.")
        # Fallback to synthetic dataset for smoke tests
        num_classes = 2
        num_samples = 1200
        base = FakeImageDataset(num_samples=num_samples, num_classes=num_classes, image_size=image_size, transform=train_tfms)
        val_ds = base.clone_with_transform(val_tfms)
        return base, val_ds, num_classes

    full_ds = ImageFolderWithIndex(root=root, transform=train_tfms)
    num_classes = len(full_ds.classes)

    # We'll return same dataset object but with different transforms applied via Subset wrappers later
    val_ds = ImageFolderWithIndex(root=root, transform=val_tfms)
    return full_ds, val_ds, num_classes


def stratified_train_val_split(indices: Sequence[int], targets: Sequence[int], train_ratio: float, val_ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    # Normalize ratios if they don't sum to 1.0
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("Sum of train_ratio and val_ratio must be > 0")
    train_ratio_n = train_ratio / total
    val_ratio_n = val_ratio / total

    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    targets = np.array(targets)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for cls in np.unique(targets):
        cls_mask = targets == cls
        cls_indices = indices[cls_mask]
        rng.shuffle(cls_indices)
        n_train = int(round(len(cls_indices) * train_ratio_n))
        n_train = max(1, min(n_train, len(cls_indices) - 1)) if len(cls_indices) > 1 else len(cls_indices)
        cls_train = cls_indices[:n_train].tolist()
        cls_val = cls_indices[n_train:].tolist()
        train_idx.extend(cls_train)
        val_idx.extend(cls_val)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def dirichlet_non_iid_split(targets: Sequence[int], num_clients: int, alpha: float, seed: int = 42) -> List[List[int]]:
    """Split indices non-IID across clients using Dirichlet distribution over labels.

    Args:
        targets: list/array of labels per sample
        num_clients: number of clients
        alpha: concentration parameter (smaller -> more skew)
    Returns:
        A list of length num_clients; each element is a list of sample indices assigned to that client
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    classes = np.unique(targets)
    num_samples = len(targets)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        # Sample proportions for this class across clients
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Compute split sizes
        splits = (proportions * len(cls_idx)).astype(int)
        # Adjust to ensure sum equals len(cls_idx)
        while splits.sum() < len(cls_idx):
            splits[rng.integers(0, num_clients)] += 1
        while splits.sum() > len(cls_idx):
            j = rng.integers(0, num_clients)
            if splits[j] > 0:
                splits[j] -= 1
        # Assign
        start = 0
        for client_id, count in enumerate(splits):
            if count > 0:
                client_indices[client_id].extend(cls_idx[start : start + count].tolist())
                start += count

    # Shuffle per-client indices
    for idxs in client_indices:
        rng.shuffle(idxs)

    # Guard: ensure every client gets at least 1 sample
    empty_clients = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
    if empty_clients:
        # simple round-robin allocation of random leftovers
        all_idx = np.arange(num_samples).tolist()
        rng.shuffle(all_idx)
        for i, ci in enumerate(empty_clients):
            client_indices[ci].append(all_idx[i])

    return client_indices


def build_client_loaders(
    full_train_ds: Dataset,
    full_val_ds: Dataset,
    num_clients: int,
    train_ratio: float,
    val_ratio: float,
    alpha: float,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
) -> Tuple[List[ClientData], List[int]]:
    # We expect full_train_ds and full_val_ds to mirror the same sample ordering/labels
    assert len(getattr(full_train_ds, "samples")) == len(getattr(full_val_ds, "samples"))

    indices = list(range(len(full_train_ds)))
    targets = full_train_ds.targets  # type: ignore[attr-defined]

    client_idx_lists = dirichlet_non_iid_split(targets, num_clients=num_clients, alpha=alpha, seed=seed)

    clients: List[ClientData] = []
    client_sizes: List[int] = []

    for client_id, client_indices in enumerate(client_idx_lists):
        # per-client stratified split
        client_targets = [targets[i] for i in client_indices]
        train_idx, val_idx = stratified_train_val_split(client_indices, client_targets, train_ratio, val_ratio, seed=seed + client_id)

        train_subset = Subset(full_train_ds, train_idx)
        val_subset = Subset(full_val_ds, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        clients.append(
            ClientData(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                num_train_samples=len(train_idx),
                num_val_samples=len(val_idx),
            )
        )
        client_sizes.append(len(train_idx))

    return clients, client_sizes
