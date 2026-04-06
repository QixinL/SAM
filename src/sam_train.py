import copy
import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.

    This wraps a base optimizer (typically SGD) and performs:
      1) gradient computation at current weights
      2) ascent step to perturbed weights
      3) gradient computation at perturbed weights
      4) actual optimizer update from original weights
    """

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
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if adaptive:
                    grad = grad * p.abs()

                norms.append(torch.norm(grad, p=2).to(shared_device))

        if len(norms) == 0:
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                if adaptive:
                    e_w = p.pow(2) * p.grad * scale
                else:
                    e_w = p.grad * scale

                p.add_(e_w)  # move to w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore original weights

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM does not support a single step() call here. "
            "Use first_step() and second_step()."
        )


def disable_bn_running_stats(model):
    """
    Freeze BatchNorm running mean/variance updates during the SAM second pass.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0


def enable_bn_running_stats(model):
    """
    Restore BatchNorm running-stat updates after the SAM second pass.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def train_baseline(
    model,
    train_loader,
    val_loader,
    epochs=200,
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    device="cuda",
):
    """
    Standard SGD baseline training.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state = None
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_sam(
    model,
    train_loader,
    val_loader,
    epochs=200,
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    rho=0.05,
    device="cuda",
    grad_clip=5.0,
):
    """
    BN-safe SAM training for ResNet-18.

    First pass:
      - normal forward/backward
      - BatchNorm updates running stats normally

    Second pass:
      - perturb weights using SAM
      - freeze BN running stats
      - forward/backward at perturbed weights
      - restore original weights and apply optimizer step
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        rho=rho,
        adaptive=False,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer,
        T_max=epochs
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state = None
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # ---- First pass: normal BN updates ----
            enable_bn_running_stats(model)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.first_step(zero_grad=True)

            # ---- Second pass: freeze BN running stats ----
            disable_bn_running_stats(model)

            outputs_sam = model(images)
            loss_sam = criterion(outputs_sam, labels)
            loss_sam.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.second_step(zero_grad=True)

            # restore BN behavior for next batch
            enable_bn_running_stats(model)

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history