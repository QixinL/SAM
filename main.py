"""
Run the baseline SGD experiment on CIFAR-10.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data import get_cifar10_loaders
from src.model import resnet18
from src.train import train_baseline


DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "epochs": 200,
    "lr": 0.1621948163070703,
    "momentum": 0.9285729026103532,
    "weight_decay": 0.0016196219073369704,
    "train_fraction": 0.1,
    "val_split": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def run_baseline(config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}

    torch.manual_seed(config["seed"])

    print(f"Device: {config['device']}")
    print(
        f"Epochs: {config['epochs']} | LR: {config['lr']} | "
        f"Momentum: {config['momentum']} | "
        f"Weight decay: {config['weight_decay']}"
    )

    train_loader, val_loader, test_loader, num_classes = get_cifar10_loaders(
        batch_size=config["batch_size"],
        train_fraction=config["train_fraction"],
        val_split=config["val_split"],
        seed=config["seed"],
    )
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Test batches: {len(test_loader)}\n"
    )

    model = resnet18(num_classes=num_classes)
    history = train_baseline(
        model,
        train_loader,
        val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        device=config["device"],
    )

    best_acc = max(history["val_acc"])
    print(f"\nBest val accuracy: {best_acc:.4f}")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Run the baseline CIFAR-10 experiment.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG["momentum"])
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"]
    )
    parser.add_argument(
        "--train-fraction", type=float, default=DEFAULT_CONFIG["train_fraction"]
    )
    parser.add_argument("--val-split", type=float, default=DEFAULT_CONFIG["val_split"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    return parser.parse_args()


def main():
    args = parse_args()
    run_baseline(
        {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "train_fraction": args.train_fraction,
            "val_split": args.val_split,
            "device": args.device,
        }
    )


if __name__ == "__main__":
    main()
