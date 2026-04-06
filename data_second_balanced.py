"""
Leakage-safe DATA_second split builder and dataloaders.

Rules:
1. Original DATA_second/Test.csv remains untouched and is the only held-out test set.
2. Only original DATA_second/Train.csv is used to create train/val.
3. Exact duplicate-content train images that also appear in the original test set
   are removed from the train/val pool conservatively.
4. The training CSV is balanced to a fixed target per class. If a class has fewer
   than the requested count after duplicate filtering and validation holdout, the
   training CSV is upsampled with replacement from the remaining train pool.
"""

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CSV_COLUMNS = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]


class DataSecondDataset(Dataset):
    def __init__(self, data_dir, samples, transform):
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(self.data_dir / sample["Path"]).convert("RGB")
        return self.transform(image), int(sample["ClassId"])


def _read_csv_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    for required in ["ClassId", "Path"]:
        if required not in rows[0]:
            raise ValueError(f"Missing required column '{required}' in {csv_path}")
    return rows


def _file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_counts(rows):
    return {str(class_id): count for class_id, count in sorted(Counter(int(r["ClassId"]) for r in rows).items())}


def _path_set(rows):
    return {row["Path"] for row in rows}


def _write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _train_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: (tensor + 0.03 * torch.randn_like(tensor)).clamp(0.0, 1.0)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_clean_split(
    data_dir="DATA_second",
    output_dir=None,
    train_target_per_class=270,
    val_ratio=0.1,
    seed=42,
    remove_exact_test_duplicates=True,
):
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / f"clean_split_seed{seed}_train{train_target_per_class}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_csv_rows(data_dir / "Train.csv")
    test_rows = _read_csv_rows(data_dir / "Test.csv")

    test_hash_to_rows = defaultdict(list)
    for row in test_rows:
        image_hash = _file_md5(data_dir / row["Path"])
        row["_hash"] = image_hash
        test_hash_to_rows[image_hash].append(row)

    duplicate_train_rows = []
    clean_train_candidates = []
    for row in train_rows:
        image_hash = _file_md5(data_dir / row["Path"])
        row["_hash"] = image_hash
        if remove_exact_test_duplicates and image_hash in test_hash_to_rows:
            duplicate_train_rows.append(
                {
                    "train_path": row["Path"],
                    "train_class_id": int(row["ClassId"]),
                    "test_matches": [
                        {"path": match["Path"], "class_id": int(match["ClassId"])}
                        for match in test_hash_to_rows[image_hash]
                    ],
                    "hash": image_hash,
                }
            )
        else:
            clean_train_candidates.append(row)

    by_class = defaultdict(list)
    for row in clean_train_candidates:
        by_class[int(row["ClassId"])].append(row)

    if len(by_class) != len(Counter(int(row["ClassId"]) for row in test_rows)):
        raise ValueError("Class mismatch between filtered train pool and original test set.")

    rng = random.Random(seed)
    train_clean_rows = []
    val_clean_rows = []
    class_report = {}

    for class_id in sorted(by_class):
        class_rows = list(by_class[class_id])
        rng.shuffle(class_rows)

        val_count = max(1, int(round(len(class_rows) * val_ratio)))
        if val_count >= len(class_rows):
            raise ValueError(
                f"Class {class_id} has too few train rows after filtering to create a validation split."
            )

        val_rows = class_rows[:val_count]
        train_pool = class_rows[val_count:]
        if not train_pool:
            raise ValueError(f"Class {class_id} has no training rows left after validation holdout.")

        if len(train_pool) >= train_target_per_class:
            selected_train = rng.sample(train_pool, train_target_per_class)
            sampling_mode = "downsample_without_replacement"
            sampled_duplicates = 0
        else:
            selected_train = list(train_pool)
            needed = train_target_per_class - len(train_pool)
            extra = [rng.choice(train_pool) for _ in range(needed)]
            selected_train.extend(extra)
            sampling_mode = "upsample_with_replacement"
            sampled_duplicates = needed

        train_clean_rows.extend(selected_train)
        val_clean_rows.extend(val_rows)
        class_report[str(class_id)] = {
            "original_train_count": sum(1 for row in train_rows if int(row["ClassId"]) == class_id),
            "filtered_train_count": len(class_rows),
            "val_count": len(val_rows),
            "train_pool_count": len(train_pool),
            "final_train_count": len(selected_train),
            "sampling_mode": sampling_mode,
            "upsampled_examples": sampled_duplicates,
            "original_test_count": sum(1 for row in test_rows if int(row["ClassId"]) == class_id),
        }

    rng.shuffle(train_clean_rows)
    rng.shuffle(val_clean_rows)

    train_clean_path = output_dir / "train_clean.csv"
    val_clean_path = output_dir / "val_clean.csv"
    test_clean_path = output_dir / "test_clean.csv"

    _write_csv(train_clean_rows, train_clean_path)
    _write_csv(val_clean_rows, val_clean_path)
    _write_csv(test_rows, test_clean_path)

    train_paths = _path_set(train_clean_rows)
    val_paths = _path_set(val_clean_rows)
    test_paths = _path_set(test_rows)

    overlap_report = {
        "train_val_path_overlap_count": len(train_paths & val_paths),
        "train_test_path_overlap_count": len(train_paths & test_paths),
        "val_test_path_overlap_count": len(val_paths & test_paths),
    }
    if any(overlap_report.values()):
        raise ValueError(f"Path overlap detected in clean split: {overlap_report}")

    train_hashes = {_file_md5(data_dir / row["Path"]) for row in train_clean_rows}
    val_hashes = {_file_md5(data_dir / row["Path"]) for row in val_clean_rows}
    test_hashes = {_file_md5(data_dir / row["Path"]) for row in test_rows}
    hash_overlap_report = {
        "train_val_hash_overlap_count": len(train_hashes & val_hashes),
        "train_test_hash_overlap_count": len(train_hashes & test_hashes),
        "val_test_hash_overlap_count": len(val_hashes & test_hashes),
    }

    report = {
        "data_dir": str(data_dir),
        "seed": seed,
        "train_target_per_class": train_target_per_class,
        "val_ratio": val_ratio,
        "train_clean_csv": str(train_clean_path),
        "val_clean_csv": str(val_clean_path),
        "test_clean_csv": str(test_clean_path),
        "num_classes": len(class_report),
        "counts": {
            "original_train_total": len(train_rows),
            "original_test_total": len(test_rows),
            "filtered_train_total": len(clean_train_candidates),
            "removed_train_duplicates_vs_test": len(duplicate_train_rows),
            "train_clean_total": len(train_clean_rows),
            "val_clean_total": len(val_clean_rows),
            "test_clean_total": len(test_rows),
        },
        "class_report": class_report,
        "path_overlap_checks": overlap_report,
        "hash_overlap_checks": hash_overlap_report,
        "exact_duplicate_train_rows_removed": duplicate_train_rows,
        "original_train_class_counts": _row_counts(train_rows),
        "original_test_class_counts": _row_counts(test_rows),
        "train_clean_class_counts": _row_counts(train_clean_rows),
        "val_clean_class_counts": _row_counts(val_clean_rows),
        "test_clean_class_counts": _row_counts(test_rows),
    }
    (output_dir / "clean_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_clean_split(split_dir):
    split_dir = Path(split_dir)
    train_rows = _read_csv_rows(split_dir / "train_clean.csv")
    val_rows = _read_csv_rows(split_dir / "val_clean.csv")
    test_rows = _read_csv_rows(split_dir / "test_clean.csv")
    report = json.loads((split_dir / "clean_split_report.json").read_text(encoding="utf-8"))
    return train_rows, val_rows, test_rows, report


def create_loaders(
    data_dir="DATA_second",
    split_dir=None,
    train_target_per_class=270,
    val_ratio=0.1,
    seed=42,
    batch_size=64,
    image_size=224,
    num_workers=0,
    include_test=True,
):
    if split_dir is None:
        split_dir = Path(data_dir) / f"clean_split_seed{seed}_train{train_target_per_class}"
    else:
        split_dir = Path(split_dir)

    if not (split_dir / "train_clean.csv").exists():
        build_clean_split(
            data_dir=data_dir,
            output_dir=split_dir,
            train_target_per_class=train_target_per_class,
            val_ratio=val_ratio,
            seed=seed,
        )

    train_rows, val_rows, test_rows, report = load_clean_split(split_dir)
    train_dataset = DataSecondDataset(data_dir, train_rows, _train_transform(image_size))
    val_dataset = DataSecondDataset(data_dir, val_rows, _eval_transform(image_size))
    test_dataset = None
    if include_test:
        test_dataset = DataSecondDataset(data_dir, test_rows, _eval_transform(image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if include_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader, test_loader, report["num_classes"], report


if __name__ == "__main__":
    report = build_clean_split()
    print(json.dumps(report["counts"], indent=2))
