"""
Balanced DATA_second loading utilities with reproducible 80/10/10 splits.
"""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DataSecondDataset(Dataset):
    def __init__(self, data_dir, samples, transform):
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.data_dir / sample["path"]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), sample["class_id"]


def _read_csv_samples(csv_path):
    samples = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                {
                    "path": row["Path"].replace("\\", "/"),
                    "class_id": int(row["ClassId"]),
                }
            )
    return samples


def _gather_samples(data_dir):
    data_dir = Path(data_dir)
    train_samples = _read_csv_samples(data_dir / "Train.csv")
    test_samples = _read_csv_samples(data_dir / "Test.csv")

    by_class = defaultdict(list)
    for sample in train_samples + test_samples:
        by_class[sample["class_id"]].append(sample)

    if not by_class:
        raise ValueError(f"No labeled samples found under {data_dir}")

    return by_class


def _split_counts(samples_per_class, train_ratio, val_ratio, test_ratio):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_count = int(samples_per_class * train_ratio)
    val_count = int(samples_per_class * val_ratio)
    test_count = samples_per_class - train_count - val_count

    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "samples_per_class is too small for the requested 80/10/10 split. "
            "Use at least 10 samples per class."
        )

    return train_count, val_count, test_count


def create_balanced_split(
    data_dir="DATA_second",
    samples_per_class=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    by_class = _gather_samples(data_dir)
    min_count = min(len(samples) for samples in by_class.values())

    if samples_per_class is None:
        samples_per_class = min_count

    if samples_per_class > min_count:
        raise ValueError(
            f"Requested {samples_per_class} samples per class, but the smallest class only has {min_count}."
        )

    train_count, val_count, test_count = _split_counts(
        samples_per_class, train_ratio, val_ratio, test_ratio
    )

    rng = random.Random(seed)
    split = {
        "data_dir": str(Path(data_dir)),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_classes": len(by_class),
        "train": [],
        "val": [],
        "test": [],
        "class_counts": {},
    }

    for class_id in sorted(by_class):
        class_samples = list(by_class[class_id])
        rng.shuffle(class_samples)
        selected = class_samples[:samples_per_class]

        split["train"].extend(selected[:train_count])
        split["val"].extend(selected[train_count:train_count + val_count])
        split["test"].extend(selected[train_count + val_count:train_count + val_count + test_count])

        split["class_counts"][str(class_id)] = {
            "selected": samples_per_class,
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }

    rng.shuffle(split["train"])
    rng.shuffle(split["val"])
    rng.shuffle(split["test"])
    return split


def save_split_manifest(split, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split_manifest(manifest_path):
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _train_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
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


def build_datasets_from_split(split, image_size=224):
    data_dir = split["data_dir"]
    train_dataset = DataSecondDataset(data_dir, split["train"], _train_transform(image_size))
    val_dataset = DataSecondDataset(data_dir, split["val"], _eval_transform(image_size))
    test_dataset = DataSecondDataset(data_dir, split["test"], _eval_transform(image_size))
    return train_dataset, val_dataset, test_dataset


def build_loaders_from_split(
    split,
    batch_size=64,
    image_size=224,
    num_workers=0,
):
    train_dataset, val_dataset, test_dataset = build_datasets_from_split(split, image_size=image_size)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def create_loaders(
    data_dir="DATA_second",
    split_manifest_path=None,
    samples_per_class=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    batch_size=64,
    image_size=224,
    num_workers=0,
):
    if split_manifest_path and Path(split_manifest_path).exists():
        split = load_split_manifest(split_manifest_path)
    else:
        split = create_balanced_split(
            data_dir=data_dir,
            samples_per_class=samples_per_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        if split_manifest_path:
            save_split_manifest(split, split_manifest_path)

    train_loader, val_loader, test_loader = build_loaders_from_split(
        split,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, split["num_classes"], split


if __name__ == "__main__":
    split = create_balanced_split()
    print(
        f"Classes: {split['num_classes']} | "
        f"Samples/class: {split['samples_per_class']} | "
        f"Train: {len(split['train'])} | "
        f"Val: {len(split['val'])} | "
        f"Test: {len(split['test'])}"
    )
