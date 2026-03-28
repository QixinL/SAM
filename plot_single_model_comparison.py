"""
Evaluate one SGD checkpoint and one SAM checkpoint, then plot train/test comparisons.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from evaluate_checkpoints import CHECKPOINT_EPOCH, SAVE_ROOT, checkpoint_path
from src.data import get_cifar10_loaders
from src.model import resnet18
from src.sam_train import evaluate


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
        description="Plot train/test metrics for one SGD checkpoint and one SAM checkpoint."
    )
    parser.add_argument("--save-root", default=SAVE_ROOT)
    parser.add_argument("--checkpoint-epoch", type=int, default=CHECKPOINT_EPOCH)
    parser.add_argument("--sgd-folder", default="sgd1")
    parser.add_argument("--sam-folder", default="sam1")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def load_model_metrics(folder, method, checkpoint_epoch, save_root, device, train_loader, test_loader, num_classes):
    ckpt = checkpoint_path(folder, checkpoint_epoch, save_root)
    payload = torch.load(ckpt, map_location=device)

    model = resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(payload["model_state"])
    criterion = nn.CrossEntropyLoss()

    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "method": method,
        "folder": folder,
        "checkpoint": ckpt,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def plot_train_test(metrics, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    method = metrics["method"].upper()
    method_color = COLORS["sam"] if metrics["method"] == "sam" else COLORS["sgd"]

    acc_ax = axes[0]
    acc_values = [metrics["train_acc"], metrics["test_acc"]]
    acc_ax.bar(["Train", "Test"], acc_values, color=[COLORS["train"], COLORS["test"]], edgecolor="white")
    acc_ax.plot(["Train", "Test"], acc_values, color=method_color, linewidth=2.2, marker="o")
    acc_ax.set_ylim(0.0, 1.0)
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title(f"{method} Accuracy")
    acc_ax.grid(axis="y", linestyle="--", alpha=0.65)

    loss_ax = axes[1]
    loss_values = [metrics["train_loss"], metrics["test_loss"]]
    loss_ax.bar(["Train", "Test"], loss_values, color=[COLORS["train"], COLORS["test"]], edgecolor="white")
    loss_ax.plot(["Train", "Test"], loss_values, color=method_color, linewidth=2.2, marker="o")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title(f"{method} Loss")
    loss_ax.grid(axis="y", linestyle="--", alpha=0.65)

    fig.suptitle(f"{method} Train vs Test Metrics", fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.text(
        0.5,
        0.01,
        f"Checkpoint: {metrics['folder']} at epoch {os.path.basename(metrics['checkpoint']).split('_')[-1].split('.')[0]}",
        ha="center",
        color=COLORS["ink"],
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_test_comparison(sgd_metrics, sam_metrics, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))

    acc_ax = axes[0]
    acc_ax.bar(
        ["SGD", "SAM"],
        [sgd_metrics["test_acc"], sam_metrics["test_acc"]],
        color=[COLORS["sgd"], COLORS["sam"]],
        edgecolor="white",
    )
    acc_ax.set_ylim(0.0, 1.0)
    acc_ax.set_ylabel("Test Accuracy")
    acc_ax.set_title("SAM Test vs SGD Test Accuracy")
    acc_ax.grid(axis="y", linestyle="--", alpha=0.65)

    loss_ax = axes[1]
    loss_ax.bar(
        ["SGD", "SAM"],
        [sgd_metrics["test_loss"], sam_metrics["test_loss"]],
        color=[COLORS["sgd"], COLORS["sam"]],
        edgecolor="white",
    )
    loss_ax.set_ylabel("Test Loss")
    loss_ax.set_title("SAM Test vs SGD Test Loss")
    loss_ax.grid(axis="y", linestyle="--", alpha=0.65)

    fig.suptitle("SAM vs SGD Test Metrics", fontsize=15, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    configure_style()
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, _val_loader, test_loader, num_classes = get_cifar10_loaders(
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        val_split=args.val_split,
        seed=args.seed,
    )

    sgd_metrics = load_model_metrics(
        args.sgd_folder,
        "sgd",
        args.checkpoint_epoch,
        args.save_root,
        args.device,
        train_loader,
        test_loader,
        num_classes,
    )
    sam_metrics = load_model_metrics(
        args.sam_folder,
        "sam",
        args.checkpoint_epoch,
        args.save_root,
        args.device,
        train_loader,
        test_loader,
        num_classes,
    )

    sgd_plot = os.path.join(args.output_dir, f"{args.sgd_folder}_train_test_{args.checkpoint_epoch:04d}.png")
    sam_plot = os.path.join(args.output_dir, f"{args.sam_folder}_train_test_{args.checkpoint_epoch:04d}.png")
    test_plot = os.path.join(args.output_dir, f"sam_vs_sgd_test_{args.checkpoint_epoch:04d}.png")
    summary_path = os.path.join(args.output_dir, f"single_model_metrics_{args.checkpoint_epoch:04d}.json")

    plot_train_test(sgd_metrics, sgd_plot)
    plot_train_test(sam_metrics, sam_plot)
    plot_test_comparison(sgd_metrics, sam_metrics, test_plot)

    summary = {"sgd": sgd_metrics, "sam": sam_metrics}
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved plot: {sgd_plot}")
    print(f"Saved plot: {sam_plot}")
    print(f"Saved plot: {test_plot}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
