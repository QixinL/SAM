"""
Baseline training loop: standard SGD with cosine LR decay.
"""
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        total_loss += criterion(outputs, labels).item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_baseline(model, train_loader, val_loader, epochs=200, lr=0.1,
                   momentum=0.9, weight_decay=0.0005, device='cuda'):
    """
    Train with standard SGD + cosine LR decay.

    Returns:
        history: dict with lists for train_loss, val_loss, val_acc per epoch
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                    weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'error_rate': []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['error_rate'].append((1 - val_acc) * 100)

        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.5f}")

    return history
