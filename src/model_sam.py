"""
Run the SAM experiment.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import DEFAULT_CONFIG
from src.data import get_cifar10_loaders
from src.model import resnet18
from src.sam_train import train_sam


SAM_CONFIG = {
    **DEFAULT_CONFIG,
    "rho": 0.05,
}


def run_sam(config=None):
    config = {**SAM_CONFIG, **(config or {})}

    torch.manual_seed(config["seed"])

    print(f"Device: {config['device']}")
    print(
        f"Epochs: {config['epochs']} | LR: {config['lr']} | "
        f"Momentum: {config['momentum']} | "
        f"Weight decay: {config['weight_decay']} | "
        f"Rho: {config['rho']}"
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
    history = train_sam(
        model,
        train_loader,
        val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        rho=config["rho"],
        device=config["device"],
    )

    best_acc = max(history["val_acc"])
    print(f"\nBest val accuracy: {best_acc:.4f}")
    return history
