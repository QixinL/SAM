"""
Quick training test: baseline model for 10 epochs.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import resnet18
from src.data import get_cifar10_loaders
from src.train import train_baseline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED   = 42

torch.manual_seed(SEED)

print(f"Device: {DEVICE}\n")

train_loader, val_loader, test_loader, num_classes = get_cifar10_loaders(
    batch_size=256,
    train_fraction=0.1,
    val_split=0.1,
    seed=SEED,
)

model = resnet18(num_classes=num_classes)

history = train_baseline(
    model, train_loader, val_loader,
    epochs=10,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    device=DEVICE,
)

print("\n── Results ──────────────────────────────────────────────")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>7}  {'Error %':>7}")
for i, (tl, vl, va) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc']), 1):
    print(f"{i:>5}  {tl:>10.4f}  {vl:>8.4f}  {va:>7.4f}  {(1 - va) * 100:>6.2f}%")

best_acc = max(history['val_acc'])
print(f"\nBest val accuracy: {best_acc:.4f}  (error: {(1 - best_acc) * 100:.2f}%)")
