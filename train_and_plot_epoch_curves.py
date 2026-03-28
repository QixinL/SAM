"""
Train SGD and SAM, evaluate on the test set every epoch, and plot metric curves.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from main import DEFAULT_CONFIG
from src.data import get_cifar10_loaders
from src.model import resnet18
from src.sam_train import SAM, disable_bn_running_stats, enable_bn_running_stats, evaluate


COLORS = {
    "sgd": "#355C7D",
    "sam": "#C06C84",
    "train": "#2A9D8F",
    "test": "#E76F51",
    "ink": "#1F2933",
    "grid": "#D8D8D8",
    "bg": "#F7F3EE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SGD and SAM and plot train/test curves per epoch."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG["momentum"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_CONFIG["train_fraction"])
    parser.add_argument("--val-split", type=float, default=DEFAULT_CONFIG["val_split"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--output-dir", default="plots")
    return parser.parse_args()


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": "white",
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "savefig.bbox": "tight",
        }
    )


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_outputs(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean().item()


def train_epoch_sgd(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def train_epoch_sam(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        enable_bn_running_stats(model)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.first_step(zero_grad=True)

        disable_bn_running_stats(model)
        outputs_sam = model(images)
        loss_sam = criterion(outputs_sam, labels)
        loss_sam.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.second_step(zero_grad=True)
        enable_bn_running_stats(model)

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def train_method(method, train_loader, test_loader, num_classes, args):
    model = resnet18(num_classes=num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()

    if method == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rho=args.rho,
            adaptive=False,
        )
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs)

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        if method == "sgd":
            train_loss, train_acc = train_epoch_sgd(model, train_loader, optimizer, criterion, args.device)
        else:
            train_loss, train_acc = train_epoch_sam(model, train_loader, optimizer, criterion, args.device, args.grad_clip)

        test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
        scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"{method.upper()} epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    return history


def plot_single_method(history, method, output_path):
    color = COLORS["sam"] if method == "sam" else COLORS["sgd"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(history["epochs"], history["train_acc"], color=COLORS["train"], linewidth=2.0, label="Train Acc")
    axes[0].plot(history["epochs"], history["test_acc"], color=COLORS["test"], linewidth=2.0, label="Test Acc")
    axes[0].set_title(f"{method.upper()} Train vs Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(linestyle="--", alpha=0.65)
    axes[0].legend(frameon=True)

    axes[1].plot(history["epochs"], history["train_loss"], color=COLORS["train"], linewidth=2.0, label="Train Loss")
    axes[1].plot(history["epochs"], history["test_loss"], color=COLORS["test"], linewidth=2.0, label="Test Loss")
    axes[1].set_title(f"{method.upper()} Train vs Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(linestyle="--", alpha=0.65)
    axes[1].legend(frameon=True)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_test_overlay(sgd_history, sam_history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(sgd_history["epochs"], sgd_history["test_acc"], color=COLORS["sgd"], linewidth=2.2, label="SGD Test Acc")
    axes[0].plot(sam_history["epochs"], sam_history["test_acc"], color=COLORS["sam"], linewidth=2.2, label="SAM Test Acc")
    axes[0].set_title("SAM Test vs SGD Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(linestyle="--", alpha=0.65)
    axes[0].legend(frameon=True)

    axes[1].plot(sgd_history["epochs"], sgd_history["test_loss"], color=COLORS["sgd"], linewidth=2.2, label="SGD Test Loss")
    axes[1].plot(sam_history["epochs"], sam_history["test_loss"], color=COLORS["sam"], linewidth=2.2, label="SAM Test Loss")
    axes[1].set_title("SAM Test vs SGD Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(linestyle="--", alpha=0.65)
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    configure_style()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_loader, _val_loader, test_loader, num_classes = get_cifar10_loaders(
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        val_split=args.val_split,
        seed=args.seed,
    )

    sgd_history = train_method("sgd", train_loader, test_loader, num_classes, args)
    sam_history = train_method("sam", train_loader, test_loader, num_classes, args)

    sgd_plot = os.path.join(args.output_dir, "sgd_epoch_curves.png")
    sam_plot = os.path.join(args.output_dir, "sam_epoch_curves.png")
    overlay_plot = os.path.join(args.output_dir, "sam_vs_sgd_test_curves.png")
    summary_path = os.path.join(args.output_dir, "epoch_curve_metrics.json")

    plot_single_method(sgd_history, "sgd", sgd_plot)
    plot_single_method(sam_history, "sam", sam_plot)
    plot_test_overlay(sgd_history, sam_history, overlay_plot)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"sgd": sgd_history, "sam": sam_history}, handle, indent=2)

    print(f"Saved plot: {sgd_plot}")
    print(f"Saved plot: {sam_plot}")
    print(f"Saved plot: {overlay_plot}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
