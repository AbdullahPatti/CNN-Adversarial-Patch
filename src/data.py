"""Dataset loaders: CIFAR-10 (main setting), GTSRB and ImageNette (second settings, J5 / J6).

Images are returned in [0, 1]; normalization is inside the model (mean / std from DATASETS).
Every config may name a `dataset` (default cifar10); CIFAR-10 behaviour is unchanged from Parts 1-5.

GTSRB: torchvision's split (GTSRB-Training_fixed: 26,640 training images; the official 12,630 test images; 43
classes, imbalanced 150-1,500 per class), each image resized whole (with its ~10% border, no ROI crop)
to 32 x 32 and cached once as uint8 (data/gtsrb/gtsrb32.pt). Augmentation is random crop only: no horizontal flip,
since flipping changes the meaning of many signs (keep left / keep right, curves).
ImageNette: imagenette2-320 (10 ImageNet classes; 9,469 train / 3,925 val, the val split is the test set), 224 px:
random-resized crop + flip for training, resize 256 + center crop 224 for testing.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

EVAL_SUBSET_SIZE = 1000  # fixed subset for LaVAN and PatchCleanser certification (plan, Sec. 1)
EVAL_SUBSET_SEED = 0

DATASETS = {
    "cifar10": {"num_classes": 10, "img": 32, "test_size": 10000,
                "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "gtsrb": {"num_classes": 43, "img": 32, "test_size": 12630,
              "mean": (0.3417, 0.3126, 0.3216), "std": (0.2737, 0.2607, 0.2662)},  # of the 32 px training set
    "imagenette": {"num_classes": 10, "img": 224, "test_size": 3925,
                   "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def spec(dataset="cifar10"):
    return DATASETS[dataset]


class _UInt8Images(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].float().div(255)
        return (self.transform(img) if self.transform else img), int(self.y[i])


def _gtsrb_cache(root):
    path = Path(root, "gtsrb", "gtsrb32.pt")
    if not path.exists():
        resize = transforms.Compose([transforms.Resize((32, 32), antialias=True), transforms.PILToTensor()])
        out = {}
        for split in ("train", "test"):
            ds = datasets.GTSRB(root, split=split, transform=resize)
            xs, ys = zip(*(ds[i] for i in range(len(ds))))
            out[split] = (torch.stack(xs), torch.tensor(ys))
        torch.save(out, path)
    return torch.load(path)


def _datasets(root, dataset):
    if dataset == "cifar10":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        return (datasets.CIFAR10(root, train=True, download=True, transform=train_tf),
                datasets.CIFAR10(root, train=False, download=True, transform=transforms.ToTensor()))
    if dataset == "gtsrb":
        c = _gtsrb_cache(root)
        return _UInt8Images(*c["train"], transforms.RandomCrop(32, padding=4)), _UInt8Images(*c["test"])
    if dataset == "imagenette":
        base = Path(root, "imagenette", "imagenette2-320")
        train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.35, 1.0)),
                                       transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        return datasets.ImageFolder(base / "train", train_tf), datasets.ImageFolder(base / "val", test_tf)
    raise ValueError(f"unknown dataset: {dataset}")


def get_loaders(root="./data", batch_size=128, num_workers=4, dataset="cifar10"):
    train, test = _datasets(root, dataset)
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def eval_subset_indices(dataset="cifar10"):
    """Same 1,000 test indices on every machine, independent of the global seed."""
    g = torch.Generator().manual_seed(EVAL_SUBSET_SEED)
    return torch.randperm(spec(dataset)["test_size"], generator=g)[:EVAL_SUBSET_SIZE].sort().values.tolist()


def get_eval_subset_loader(root="./data", batch_size=100, num_workers=4, dataset="cifar10"):
    test = _datasets(root, dataset)[1]
    return DataLoader(Subset(test, eval_subset_indices(dataset)), batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
