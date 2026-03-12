"""
Load CIFAR-10 dataset with augmentation.
"""
import platform
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset


class _TransformSubset(Dataset):
    """Wraps a Subset and applies a specific transform, ignoring the base dataset's transform."""
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        return self.transform(img), label


def get_cifar10_loaders(batch_size=256, train_fraction=0.1, val_split=0.1, seed=42):
    """
    Load CIFAR-10 training and test data.

    - train_loader: augmented (RandomCrop + RandomHorizontalFlip)
    - val_loader:   no augmentation
    - test_loader:  no augmentation

    Args:
        batch_size: Batch size
        train_fraction: Use only this fraction of training data (for small experiments)
        val_split: Split this fraction from training for validation
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load the raw training set without any transform so we can apply
    # train_transform or test_transform to specific splits ourselves
    raw_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    test_set   = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Pick a random subset of the full training set
    n_total  = len(raw_train)
    n_subset = int(n_total * train_fraction)
    n_val    = int(n_subset * val_split)
    n_train  = n_subset - n_val

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=rng).tolist()

    train_indices = perm[:n_train]
    val_indices   = perm[n_train:n_train + n_val]

    train_set = _TransformSubset(raw_train, train_indices, train_transform)
    val_set   = _TransformSubset(raw_train, val_indices,   test_transform)

    # num_workers > 0 causes hangs on Windows; pin_memory speeds up CPU->GPU transfers
    num_workers = 0 if platform.system() == 'Windows' else 2

    # Create loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, 10
