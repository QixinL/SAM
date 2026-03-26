"""
Train SGD and SAM replicas with resumable checkpoints.

Saves checkpoints every N epochs into clean run folders, e.g.:
- saves/sgd1
- saves/sam1

Use this script only for training and checkpoint generation.
Use evaluate_checkpoints.py to compute metrics from saved checkpoints.
"""

import argparse
import json
import os
import random
import re
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from main import DEFAULT_CONFIG
from src.data import get_cifar10_loaders
from src.model import resnet18
from src.sam_train import SAM, disable_bn_running_stats, enable_bn_running_stats


_CKPT_PATTERN = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def checkpoint_path(run_dir, epoch):
    return os.path.join(run_dir, f"checkpoint_epoch_{epoch:04d}.pt")


def latest_checkpoint(run_dir):
    if not os.path.isdir(run_dir):
        return None

    best_epoch = -1
    best_path = None

    for name in os.listdir(run_dir):
        match = _CKPT_PATTERN.match(name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = os.path.join(run_dir, name)

    return best_path


def save_checkpoint(
    run_dir,
    epoch,
    model,
    optimizer,
    scheduler,
    method,
    seed,
    config,
    save_every,
):
    payload = {
        "epoch": epoch,
        "method": method,
        "seed": seed,
        "save_every": save_every,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }

    path = checkpoint_path(run_dir, epoch)
    torch.save(payload, path)
    torch.save(payload, os.path.join(run_dir, "latest.pt"))


def load_checkpoint(checkpoint_file, model, optimizer, scheduler, device):
    payload = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    return int(payload["epoch"])


def train_one_run(method, replica_id, seed, args):
    run_dir = os.path.join(args.save_root, f"{method}{replica_id}")
    ensure_dir(run_dir)

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "rho": args.rho,
        "grad_clip": args.grad_clip,
        "train_fraction": args.train_fraction,
        "val_split": args.val_split,
        "device": args.device,
        "max_epochs": args.max_epochs,
    }

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"method": method, "replica_id": replica_id, "seed": seed, "config": config}, f, indent=2)

    set_seed(seed)

    train_loader, _val_loader, _test_loader, num_classes = get_cifar10_loaders(
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        val_split=args.val_split,
        seed=seed,
    )

    model = resnet18(num_classes=num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()

    if method == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    elif method == "sam":
        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rho=args.rho,
            adaptive=False,
        )
        scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=args.max_epochs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    start_epoch = 1
    resume_file = latest_checkpoint(run_dir) if args.resume else None
    if resume_file is not None:
        done_epoch = load_checkpoint(resume_file, model, optimizer, scheduler, args.device)
        start_epoch = done_epoch + 1
        print(f"Resuming {method}{replica_id} from epoch {done_epoch}: {resume_file}")

    if start_epoch > args.max_epochs:
        print(f"Skipping {method}{replica_id}: already completed to epoch {args.max_epochs}")
        return

    print(f"Training {method}{replica_id} from epoch {start_epoch} to {args.max_epochs}")

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()

        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            if method == "sgd":
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                enable_bn_running_stats(model)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.first_step(zero_grad=True)

                disable_bn_running_stats(model)
                outputs_sam = model(images)
                loss_sam = criterion(outputs_sam, labels)
                loss_sam.backward()

                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.second_step(zero_grad=True)
                enable_bn_running_stats(model)

        scheduler.step()

        if epoch % args.save_every == 0 or epoch == args.max_epochs:
            save_checkpoint(
                run_dir=run_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                method=method,
                seed=seed,
                config=config,
                save_every=args.save_every,
            )
            print(f"  saved checkpoint at epoch {epoch}: {checkpoint_path(run_dir, epoch)}")

    final_ckpt = checkpoint_path(run_dir, args.max_epochs)
    if os.path.exists(final_ckpt):
        payload = torch.load(final_ckpt, map_location="cpu")
        torch.save(payload, os.path.join(run_dir, "final.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM/SGD replicas with frequent checkpoints.")

    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-root", default="saves")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG["momentum"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_CONFIG["train_fraction"])
    parser.add_argument("--val-split", type=float, default=DEFAULT_CONFIG["val_split"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.save_root)

    started = time.time()
    methods = ["sgd", "sam"]

    print("Training checkpoint generator")
    print(f"Methods: {methods}")
    print(f"Replicas: {args.replicas}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Save every: {args.save_every}")
    print(f"Save root: {args.save_root}")

    for method in methods:
        method_offset = 0 if method == "sgd" else 1_000_000
        for replica_idx in range(1, args.replicas + 1):
            seed = args.base_seed + method_offset + (replica_idx - 1)
            train_one_run(method=method, replica_id=replica_idx, seed=seed, args=args)

    elapsed = time.time() - started
    print(f"Done training checkpoints in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
