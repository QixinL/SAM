"""
Evaluate a baseline ResNet-18 checkpoint on the balanced DATA_second split.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from data_second_balanced import create_loaders


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline ResNet-18 on DATA_second.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    _, val_loader, test_loader, num_classes, split = create_loaders(
        split_manifest_path=args.split_manifest,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)

    results = {
        "checkpoint": str(Path(args.checkpoint)),
        "split_manifest": str(Path(args.split_manifest)),
        "num_classes": num_classes,
        "samples_per_class": split["samples_per_class"],
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
