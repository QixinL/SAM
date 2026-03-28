"""
Plot checkpoint replica results as explicit points with mean and 95% CI.
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
        description="Plot SAM and SGD checkpoint accuracies as replica points."
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


def extract_values(results):
    return [result["test_acc"] for result in valid_results(results)]


def draw_group(ax, label, values, color):
    x_values = list(range(1, len(values) + 1))
    mu = mean(values)
    ci_low, ci_high = ci95(values)

    ax.scatter(x_values, values, s=90, color=color, edgecolors="white", linewidths=1.2, zorder=3)
    ax.axhline(mu, color=COLORS["ink"], linewidth=2.0, linestyle="--", zorder=2)
    ax.axhspan(ci_low, ci_high, color=color, alpha=0.14, zorder=1)

    for x_value, value in zip(x_values, values):
        ax.text(x_value, value + 0.0015, f"{value:.4f}", ha="center", va="bottom", fontsize=8, color=COLORS["ink"])

    ymin = max(0.0, min(values + [ci_low]) - 0.01)
    ymax = min(1.0, max(values + [ci_high]) + 0.01)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_values)
    ax.set_xlabel("Replica")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"{label} Replica Test Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    ax.text(
        0.02,
        0.96,
        f"mean={mu:.4f}\n95% CI=[{ci_low:.4f}, {ci_high:.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )


def draw_overlay(ax, sgd_values, sam_values):
    sgd_x = list(range(1, len(sgd_values) + 1))
    sam_x = list(range(1, len(sam_values) + 1))
    sgd_mean = mean(sgd_values)
    sam_mean = mean(sam_values)
    sgd_ci_low, sgd_ci_high = ci95(sgd_values)
    sam_ci_low, sam_ci_high = ci95(sam_values)

    ax.scatter(sgd_x, sgd_values, s=90, color=COLORS["sgd"], edgecolors="white", linewidths=1.2, label="SGD", zorder=3)
    ax.scatter(sam_x, sam_values, s=90, color=COLORS["sam"], edgecolors="white", linewidths=1.2, label="SAM", marker="s", zorder=3)
    ax.axhspan(sgd_ci_low, sgd_ci_high, color=COLORS["sgd"], alpha=0.1, zorder=1)
    ax.axhspan(sam_ci_low, sam_ci_high, color=COLORS["sam"], alpha=0.1, zorder=1)
    ax.axhline(sgd_mean, color=COLORS["sgd"], linewidth=2.0, linestyle="--", zorder=2)
    ax.axhline(sam_mean, color=COLORS["sam"], linewidth=2.0, linestyle="--", zorder=2)

    combined = sgd_values + sam_values + [sgd_ci_low, sgd_ci_high, sam_ci_low, sam_ci_high]
    ax.set_ylim(max(0.0, min(combined) - 0.01), min(1.0, max(combined) + 0.01))
    ax.set_xticks(sgd_x)
    ax.set_xlabel("Replica")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("SAM and SGD Replica Overlay")
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    ax.legend(frameon=True)


def main():
    args = parse_args()
    configure_style()
    os.makedirs(args.output_dir, exist_ok=True)

    sgd_results = evaluate_folders(SGD_FOLDERS, args.checkpoint_epoch, save_root=args.save_root)
    sam_results = evaluate_folders(SAM_FOLDERS, args.checkpoint_epoch, save_root=args.save_root)
    sgd_values = extract_values(sgd_results)
    sam_values = extract_values(sam_results)

    summary = {
        "checkpoint_epoch": args.checkpoint_epoch,
        "sgd": summarize_results("sgd", sgd_results),
        "sam": summarize_results("sam", sam_results),
    }
    summary_path = os.path.join(args.output_dir, f"checkpoint_points_summary_{args.checkpoint_epoch:04d}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    sgd_plot = os.path.join(args.output_dir, f"sgd_points_{args.checkpoint_epoch:04d}.png")
    sam_plot = os.path.join(args.output_dir, f"sam_points_{args.checkpoint_epoch:04d}.png")
    overlay_plot = os.path.join(args.output_dir, f"sam_sgd_points_overlay_{args.checkpoint_epoch:04d}.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    draw_group(ax, "SGD", sgd_values, COLORS["sgd"])
    fig.savefig(sgd_plot, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    draw_group(ax, "SAM", sam_values, COLORS["sam"])
    fig.savefig(sam_plot, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    draw_overlay(ax, sgd_values, sam_values)
    fig.savefig(overlay_plot, dpi=220)
    plt.close(fig)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {sgd_plot}")
    print(f"Saved plot: {sam_plot}")
    print(f"Saved plot: {overlay_plot}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
