"""
Plot per-class train/test image counts for the DATA_second dataset.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLORS = {
    "train": "#355C7D",
    "test": "#E56B6F",
    "ink": "#1F2933",
    "grid": "#D8D8D8",
    "bg": "#F7F3EE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot class distribution for DATA_second."
    )
    parser.add_argument("--data-dir", default="DATA_second")
    parser.add_argument("--output-dir", default="plot_data")
    parser.add_argument(
        "--output-name", default="data_second_class_distribution.png"
    )
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
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "savefig.bbox": "tight",
        }
    )


def count_by_class(csv_path: Path) -> Counter:
    counts = Counter()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[int(row["ClassId"])] += 1
    return counts


def build_rows(train_counts: Counter, test_counts: Counter):
    class_ids = sorted(set(train_counts) | set(test_counts))
    rows = []
    for class_id in class_ids:
        train = train_counts.get(class_id, 0)
        test = test_counts.get(class_id, 0)
        rows.append(
            {
                "class_id": class_id,
                "train": train,
                "test": test,
                "total": train + test,
            }
        )
    return rows


def plot_distribution(rows, output_path: Path):
    class_ids = [row["class_id"] for row in rows]
    train_counts = [row["train"] for row in rows]
    test_counts = [row["test"] for row in rows]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(class_ids, train_counts, color=COLORS["train"], label="Train", width=0.82)
    ax.bar(
        class_ids,
        test_counts,
        bottom=train_counts,
        color=COLORS["test"],
        label="Test",
        width=0.82,
    )

    ax.set_title("DATA_second Class Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Image Count")
    ax.set_xticks(class_ids)
    ax.set_xlim(min(class_ids) - 0.8, max(class_ids) + 0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    total_train = sum(train_counts)
    total_test = sum(test_counts)
    total_images = total_train + total_test
    ax.text(
        0.99,
        0.98,
        f"Train: {total_train:,}\nTest: {total_test:,}\nTotal: {total_images:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 8},
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    configure_style()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts = count_by_class(data_dir / "Train.csv")
    test_counts = count_by_class(data_dir / "Test.csv")
    rows = build_rows(train_counts, test_counts)

    output_path = output_dir / args.output_name
    plot_distribution(rows, output_path)

    summary = {
        "dataset": str(data_dir),
        "train_total": sum(train_counts.values()),
        "test_total": sum(test_counts.values()),
        "all_total": sum(train_counts.values()) + sum(test_counts.values()),
        "classes": rows,
        "plot_path": str(output_path),
    }
    summary_path = output_dir / "data_second_class_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved plot to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
