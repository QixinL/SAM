"""
Run the SAM experiment.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import resnet18
from src.data import get_cifar10_loaders
from src.train import train_sam

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 256
EPOCHS        = 200
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0005
RHO           = 0.05
TRAIN_FRAC    = 0.1   # 10% of CIFAR-10
VAL_SPLIT     = 0.1   # 10% of that as validation
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(
    f"Epochs: {EPOCHS} | LR: {LR} | Momentum: {MOMENTUM} | "
    f"Weight decay: {WEIGHT_DECAY} | Rho: {RHO}"
)

train_loader, val_loader, test_loader, num_classes = get_cifar10_loaders(
    batch_size=BATCH_SIZE,
    train_fraction=TRAIN_FRAC,
    val_split=VAL_SPLIT,
    seed=SEED,
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

model = resnet18(num_classes=num_classes)

history = train_sam(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    rho=RHO,
    device=DEVICE,
)

best_acc = max(history["val_acc"])
print(f"\nBest val accuracy: {best_acc:.4f}")