from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainResult:
    num_samples: int
    train_loss: float


@dataclass
class EvalResult:
    num_samples: int
    correct: int
    loss: float


def train_local(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> TrainResult:
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_samples = 0

    for _ in range(epochs):
        pbar = tqdm(data_loader, leave=False, desc="local-train")
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    return TrainResult(num_samples=total_samples, train_loss=avg_loss)


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return EvalResult(num_samples=total_samples, correct=total_correct, loss=total_loss / max(1, total_samples))


def get_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def weighted_average_states(states_and_weights: Tuple[Dict[str, torch.Tensor], int] | list[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    if not isinstance(states_and_weights, list):
        states_and_weights = [states_and_weights]
    total = sum(w for _, w in states_and_weights)
    assert total > 0

    # Initialize accumulators
    keys = states_and_weights[0][0].keys()
    avg: Dict[str, torch.Tensor] = {k: torch.zeros_like(states_and_weights[0][0][k]) for k in keys}

    for state, weight in states_and_weights:
        for k in keys:
            avg[k] += state[k] * (weight / total)

    return avg
