"""
Minimal checkpoint evaluator.

Edit FOLDERS and CHECKPOINT_EPOCH, then run this file.
It returns one result per folder.
"""

import json
import math
import os
from statistics import mean, stdev

import torch
import torch.nn as nn

from main import DEFAULT_CONFIG
from src.data import get_cifar10_loaders
from src.model import resnet18
from src.sam_train import evaluate


# User inputs
SGD_FOLDERS = ["sgd1", "sgd2", "sgd3", "sgd4", "sgd5"]
SAM_FOLDERS = ["sam1", "sam2", "sam3", "sam4", "sam5"]
CHECKPOINT_EPOCH = 200

# Optional settings
SAVE_ROOT = "saves_200"
DEVICE = DEFAULT_CONFIG["device"]
BATCH_SIZE = DEFAULT_CONFIG["batch_size"]
TRAIN_FRACTION = DEFAULT_CONFIG["train_fraction"]
VAL_SPLIT = DEFAULT_CONFIG["val_split"]
SEED = DEFAULT_CONFIG["seed"]


_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def checkpoint_path(folder, checkpoint_epoch, save_root):
    return os.path.join(save_root, folder, f"checkpoint_epoch_{checkpoint_epoch:04d}.pt")


def ci95(values):
    n = len(values)
    mu = mean(values)
    if n < 2:
        return mu, mu

    s = stdev(values)
    t_value = _T_CRIT_95.get(n - 1, 1.96)
    half_width = t_value * s / math.sqrt(n)
    return mu - half_width, mu + half_width


def summarize_results(group_name, results):
    valid = [r for r in results if "test_acc" in r]
    if len(valid) == 0:
        return {
            "group": group_name,
            "n": 0,
            "error": "no valid checkpoints",
        }

    accs = [r["test_acc"] for r in valid]
    errors = [r["test_error"] for r in valid]

    acc_mean = mean(accs)
    acc_ci_low, acc_ci_high = ci95(accs)
    err_mean = mean(errors)
    err_ci_low, err_ci_high = ci95(errors)

    return {
        "group": group_name,
        "n": len(valid),
        "acc_mean": acc_mean,
        "acc_ci95": [acc_ci_low, acc_ci_high],
        "error_mean": err_mean,
        "error_ci95": [err_ci_low, err_ci_high],
    }


def evaluate_folders(
    folders,
    checkpoint_epoch,
    save_root=SAVE_ROOT,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    train_fraction=TRAIN_FRACTION,
    val_split=VAL_SPLIT,
    seed=SEED,
):
    _train_loader, _val_loader, test_loader, num_classes = get_cifar10_loaders(
        batch_size=batch_size,
        train_fraction=train_fraction,
        val_split=val_split,
        seed=seed,
    )
    criterion = nn.CrossEntropyLoss()

    results = []
    for folder in folders:
        ckpt = checkpoint_path(folder, checkpoint_epoch, save_root)
        if not os.path.exists(ckpt):
            result = {
                "folder": folder,
                "checkpoint": ckpt,
                "error": "missing checkpoint",
            }
            results.append(result)
            print(json.dumps(result, indent=2))
            continue

        payload = torch.load(ckpt, map_location=device)
        model = resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(payload["model_state"])

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        result = {
            "folder": folder,
            "checkpoint": ckpt,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_error": 1.0 - test_acc,
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    return results


if __name__ == "__main__":
    print(f"Evaluating SGD folders: {SGD_FOLDERS} at epoch: {CHECKPOINT_EPOCH}\n")
    print("Give each a moment, should take less than a minute each")
    sgd_results = evaluate_folders(SGD_FOLDERS, CHECKPOINT_EPOCH)

    print(f"\nEvaluating SAM folders: {SAM_FOLDERS} at epoch: {CHECKPOINT_EPOCH}\n")
    sam_results = evaluate_folders(SAM_FOLDERS, CHECKPOINT_EPOCH)

    summary = {
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "sgd": summarize_results("sgd", sgd_results),
        "sam": summarize_results("sam", sam_results),
    }

    print("\nFinal summary (mean + 95% CI):")
    print(json.dumps(summary, indent=2))
