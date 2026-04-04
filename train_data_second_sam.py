"""
Train a ResNet-18 with SAM on a balanced DATA_second split.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from data_second_balanced import create_loaders, save_split_manifest


DEFAULT_CONFIG = {
    "data_dir": "DATA_second",
    "output_dir": "runs/data_second_sam",
    "samples_per_class": None,
    "batch_size": 64,
    "epochs": 50,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "rho": 0.05,
    "grad_clip": 5.0,
    "image_size": 224,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if adaptive:
                    grad = grad * parameter.abs()
                norms.append(torch.norm(grad, p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            return
        for group in self.param_groups:
            rho = group["rho"]
            adaptive = group["adaptive"]
            scale = rho / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.data.clone()
                e_w = (parameter.pow(2) if adaptive else 1.0) * parameter.grad * scale
                parameter.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data = self.state[parameter]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step() and second_step() for SAM.")


def disable_bn_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0


def enable_bn_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum


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


def train(config):
    torch.manual_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split_manifest.json"

    train_loader, val_loader, test_loader, num_classes, split = create_loaders(
        data_dir=config["data_dir"],
        split_manifest_path=split_path,
        samples_per_class=config["samples_per_class"],
        seed=config["seed"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
    )
    save_split_manifest(split, split_path)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(config["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        rho=config["rho"],
        adaptive=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer, T_max=config["epochs"]
    )

    history = []
    best_val_acc = -1.0
    best_checkpoint_path = output_dir / "best_sam.pt"

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            enable_bn_running_stats(model)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if config["grad_clip"] is not None and config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.first_step(zero_grad=True)

            disable_bn_running_stats(model)
            outputs_sam = model(images)
            loss_sam = criterion(outputs_sam, labels)
            loss_sam.backward()

            if config["grad_clip"] is not None and config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.second_step(zero_grad=True)
            enable_bn_running_stats(model)

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
                    "split_manifest_path": str(split_path),
                },
                best_checkpoint_path,
            )

    test_loss, test_acc = evaluate(model, test_loader, criterion, config["device"])
    summary = {
        "config": config,
        "best_val_acc": best_val_acc,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "num_classes": num_classes,
        "split_manifest_path": str(split_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }
    (output_dir / "sam_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best checkpoint saved to {best_checkpoint_path}")
    print(f"Final test accuracy: {test_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM ResNet-18 on DATA_second.")
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_CONFIG["samples_per_class"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CONFIG["momentum"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--rho", type=float, default=DEFAULT_CONFIG["rho"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
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
            "samples_per_class": args.samples_per_class,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "rho": args.rho,
            "grad_clip": args.grad_clip,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": args.device,
        }
    )
