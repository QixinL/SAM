"""
Create replica distribution plots for SAM and SGD checkpoint evaluations.
"""

import argparse
import json
import os
from statistics import mean

import matplotlib.pyplot as plt

from evaluate_checkpoints import (
    CHECKPOINT_EPOCH,
    SAM_FOLDERS,
    SAVE_ROOT,
    SGD_FOLDERS,
    ci95,
    evaluate_folders,
    summarize_results,
)


COLORS = {
    "sam": "#C06C84",
    "sgd": "#355C7D",
    "ink": "#1F2933",
    "grid": "#D8D8D8",
    "bg": "#F7F3EE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SAM and SGD checkpoint accuracy distributions."
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=CHECKPOINT_EPOCH)
    parser.add_argument("--save-root", default=SAVE_ROOT)
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


def valid_results(results):
    return [result for result in results if "test_acc" in result]


def metric_values(results):
    return [result["test_acc"] for result in valid_results(results)]


def draw_histogram(ax, values, color, label):
    bins = max(4, min(8, len(values) + 1))
    ax.hist(values, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=1.2)

    mu = mean(values)
    ci_low, ci_high = ci95(values)
    ax.axvline(mu, color=COLORS["ink"], linewidth=2.2, linestyle="--", label=f"{label} mean")
    ax.axvspan(ci_low, ci_high, color=color, alpha=0.18, label=f"{label} 95% CI")

    y_top = ax.get_ylim()[1]
    ax.text(
        mu,
        y_top * 0.92,
        f"mean={mu:.4f}\nCI=[{ci_low:.4f}, {ci_high:.4f}]",
        ha="center",
        va="top",
        fontsize=9,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )


def save_single_histogram(values, label, color, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_histogram(ax, values, color, label)
    ax.set_title(f"{label} Test Accuracy Distribution")
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Replica Count")
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_overlay_histogram(sgd_values, sam_values, output_path):
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    bins = max(5, min(10, len(sgd_values) + len(sam_values)))

    ax.hist(
        sgd_values,
        bins=bins,
        color=COLORS["sgd"],
        alpha=0.5,
        edgecolor="white",
        linewidth=1.2,
        label="SGD",
    )
    ax.hist(
        sam_values,
        bins=bins,
        color=COLORS["sam"],
        alpha=0.5,
        edgecolor="white",
        linewidth=1.2,
        label="SAM",
    )

    sgd_mean = mean(sgd_values)
    sam_mean = mean(sam_values)
    sgd_ci_low, sgd_ci_high = ci95(sgd_values)
    sam_ci_low, sam_ci_high = ci95(sam_values)

    ax.axvline(sgd_mean, color=COLORS["sgd"], linewidth=2.4, linestyle="--")
    ax.axvline(sam_mean, color=COLORS["sam"], linewidth=2.4, linestyle="--")
    ax.axvspan(sgd_ci_low, sgd_ci_high, color=COLORS["sgd"], alpha=0.12)
    ax.axvspan(sam_ci_low, sam_ci_high, color=COLORS["sam"], alpha=0.12)

    ax.text(
        0.02,
        0.96,
        f"SGD mean={sgd_mean:.4f} | CI=[{sgd_ci_low:.4f}, {sgd_ci_high:.4f}]\n"
        f"SAM mean={sam_mean:.4f} | CI=[{sam_ci_low:.4f}, {sam_ci_high:.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )

    ax.set_title("SAM vs SGD Test Accuracy Overlay")
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Replica Count")
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    configure_style()
    os.makedirs(args.output_dir, exist_ok=True)

    sgd_results = evaluate_folders(SGD_FOLDERS, args.checkpoint_epoch, save_root=args.save_root)
    sam_results = evaluate_folders(SAM_FOLDERS, args.checkpoint_epoch, save_root=args.save_root)

    sgd_values = metric_values(sgd_results)
    sam_values = metric_values(sam_results)

    summary = {
        "checkpoint_epoch": args.checkpoint_epoch,
        "sgd": summarize_results("sgd", sgd_results),
        "sam": summarize_results("sam", sam_results),
    }
    summary_path = os.path.join(
        args.output_dir,
        f"checkpoint_distribution_summary_{args.checkpoint_epoch:04d}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    sgd_plot = os.path.join(args.output_dir, f"sgd_hist_{args.checkpoint_epoch:04d}.png")
    sam_plot = os.path.join(args.output_dir, f"sam_hist_{args.checkpoint_epoch:04d}.png")
    overlay_plot = os.path.join(args.output_dir, f"sam_sgd_overlay_{args.checkpoint_epoch:04d}.png")

    save_single_histogram(sgd_values, "SGD", COLORS["sgd"], sgd_plot)
    save_single_histogram(sam_values, "SAM", COLORS["sam"], sam_plot)
    save_overlay_histogram(sgd_values, sam_values, overlay_plot)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {sgd_plot}")
    print(f"Saved plot: {sam_plot}")
    print(f"Saved plot: {overlay_plot}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
