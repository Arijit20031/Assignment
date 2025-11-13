from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dad_federated.data import load_imagefolder_dataset, build_client_loaders
from dad_federated.models import build_model
from dad_federated.federated import federated_train
from dad_federated.utils import RunConfig, set_seed, get_device, ensure_dir, save_json, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Federated training on DAD-like dataset (ImageFolder)")
    p.add_argument("--data_root", type=str, default=str(Path("data/DAD")), help="Path to DAD dataset root (ImageFolder-like)")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for non-IID split (smaller=more skew)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--run_dir", type=str, default=str(Path("runs/fed")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    ensure_dir(args.run_dir)

    full_train_ds, full_val_ds, num_classes = load_imagefolder_dataset(args.data_root, image_size=args.image_size, use_fake_if_missing=True)

    model = build_model(args.model, num_classes=num_classes, pretrained=False)
    print(f"Model params: {count_parameters(model):,}")

    clients, client_sizes = build_client_loaders(
        full_train_ds=full_train_ds,
        full_val_ds=full_val_ds,
        num_clients=args.num_clients,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print("Client sizes (train):", client_sizes)

    history = federated_train(
        model=model,
        clients=clients,
        device=device,
        global_rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )

    cfg = RunConfig(
        data_root=args.data_root,
        num_clients=args.num_clients,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        noniid_alpha=args.alpha,
        global_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        model_name=args.model,
        image_size=args.image_size,
    )

    # Save artifacts
    ensure_dir(args.run_dir)
    save_json(Path(args.run_dir) / "config.json", cfg.to_dict())

    hist_dict = [h.__dict__ for h in history]
    save_json(Path(args.run_dir) / "history.json", {"rounds": hist_dict})

    torch.save(model.state_dict(), Path(args.run_dir) / "model_final.pt")
    print("Training complete. Artifacts saved to:", args.run_dir)


if __name__ == "__main__":
    main()
