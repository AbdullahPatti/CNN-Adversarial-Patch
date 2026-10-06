"""CIFAR-10 loaders. Images are returned in [0, 1]; normalization is inside the model."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

EVAL_SUBSET_SIZE = 1000  # fixed subset for LaVAN and PatchCleanser certification (plan, Sec. 1)
EVAL_SUBSET_SEED = 0


def get_loaders(root="./data", batch_size=128, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def eval_subset_indices():
    """Same 1,000 test indices on every machine, independent of the global seed."""
    g = torch.Generator().manual_seed(EVAL_SUBSET_SEED)
    return torch.randperm(10000, generator=g)[:EVAL_SUBSET_SIZE].sort().values.tolist()


def get_eval_subset_loader(root="./data", batch_size=100, num_workers=4):
    test = datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(Subset(test, eval_subset_indices()), batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
