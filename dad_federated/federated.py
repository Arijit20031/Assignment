from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .train import TrainResult, EvalResult, get_state_dict, set_state_dict, weighted_average_states, train_local, evaluate


@dataclass
class RoundMetrics:
    round: int
    avg_train_loss: float
    val_accuracy: float
    val_loss: float


def federated_train(
    model: torch.nn.Module,
    clients,
    device: torch.device,
    global_rounds: int,
    local_epochs: int,
    lr: float,
) -> List[RoundMetrics]:
    history: List[RoundMetrics] = []

    # Initialize global state
    global_state = get_state_dict(model)

    for r in range(1, global_rounds + 1):
        client_states: List[Dict] = []
        client_weights: List[int] = []
        train_losses: List[float] = []

        # Broadcast global state
        for client in clients:
            set_state_dict(model, global_state)
            # Train locally
            train_res: TrainResult = train_local(
                model=model,
                data_loader=client.train_loader,
                device=device,
                epochs=local_epochs,
                lr=lr,
            )
            train_losses.append(train_res.train_loss)
            client_states.append(get_state_dict(model))
            client_weights.append(client.num_train_samples)

        # Aggregate
        global_state = weighted_average_states(list(zip(client_states, client_weights)))
        set_state_dict(model, global_state)

        # Evaluate on combined validation across clients
        total_val_samples = 0
        total_val_correct = 0
        total_val_loss = 0.0

        for client in clients:
            set_state_dict(model, global_state)
            eval_res: EvalResult = evaluate(model, client.val_loader, device)
            total_val_samples += eval_res.num_samples
            total_val_correct += eval_res.correct
            total_val_loss += eval_res.loss * eval_res.num_samples

        val_accuracy = total_val_correct / max(1, total_val_samples)
        val_loss = total_val_loss / max(1, total_val_samples)
        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        history.append(
            RoundMetrics(round=r, avg_train_loss=avg_train_loss, val_accuracy=val_accuracy, val_loss=val_loss)
        )

    # restore final state to model
    set_state_dict(model, global_state)
    return history
