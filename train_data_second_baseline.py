"""
Train a ResNet-18 baseline on a leakage-safe DATA_second split.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18

from data_second_balanced import create_loaders


DEFAULT_CONFIG = {
    "data_dir": "DATA_second",
    "output_dir": "runs/data_second_baseline",
    "split_dir": None,
    "train_target_per_class": 270,
    "val_ratio": 0.1,
    "batch_size": 64,
    "epochs": 20,
    "lr": 0.019920,
    "momentum": 0.8132,
    "weight_decay": 0.000038,
    "image_size": 224,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_training_curves(history, output_path):
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, train_acc, label="Train Acc", linewidth=2.0)
    axes[0].plot(epochs, val_acc, label="Val Acc", linewidth=2.0)
    axes[0].set_title("Baseline Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, label="Train Loss", linewidth=2.0)
    axes[1].plot(epochs, val_loss, label="Val Loss", linewidth=2.0)
    axes[1].set_title("Baseline Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train(config):
    torch.manual_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, num_classes, split = create_loaders(
        data_dir=config["data_dir"],
        split_dir=config["split_dir"],
        train_target_per_class=config["train_target_per_class"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        include_test=False,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(config["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history = []
    best_val_acc = -1.0
    best_checkpoint_path = output_dir / "best_baseline.pt"
    plot_path = output_dir / "baseline_epoch_curves.png"

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, config["device"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "best_epoch": epoch,
                    "split_dir": str(Path(split["train_clean_csv"]).parent),
                },
                best_checkpoint_path,
            )

    plot_training_curves(history, plot_path)

    summary = {
        "config": config,
        "best_val_acc": best_val_acc,
        "num_classes": num_classes,
        "split_dir": str(Path(split["train_clean_csv"]).parent),
        "best_checkpoint_path": str(best_checkpoint_path),
        "plot_path": str(plot_path),
        "history": history,
        "clean_split_report": split,
    }
    (output_dir / "baseline_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint saved to {best_checkpoint_path}")
    print(f"Training curves saved to {plot_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline ResNet-18 on DATA_second.")
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--split-dir", default=DEFAULT_CONFIG["split_dir"])
    parser.add_argument("--train-target-per-class", type=int, default=DEFAULT_CONFIG["train_target_per_class"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_CONFIG["val_ratio"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG["momentum"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "split_dir": args.split_dir,
            "train_target_per_class": args.train_target_per_class,
            "val_ratio": args.val_ratio,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": args.device,
        }
    )
