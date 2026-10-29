"""ResNet-18, CIFAR variant (3x3 stem, no max-pool), as in He et al. 2016 / kuangliu/pytorch-cifar; ResNet-50
(bottleneck, ImageNet 7x7 stem + max-pool) for ImageNette at 224 px; MobileNetV2 in src/models/mobilenetv2.py.

Normalization lives inside the model so every model takes images in [0, 1].
Patch attacks and PatchCleanser masking both operate in raw pixel space, so this
keeps them model-agnostic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """width is the stem / layer1 channel count (64 = standard). Stages use w, 2w, 4w, 8w.

    Smaller widths give the KD students and the dense-small capacity control (RQ4):
    45 / 32 / 23 -> 2.02x / 3.99x / 7.72x fewer parameters than width 64.
    """

    def __init__(self, num_classes=10, width=64, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        w = width
        self.norm = Normalize(mean, std)
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        layers, in_planes = [], w
        for planes, stride in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers.append(nn.Sequential(BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes, 1)))
            in_planes = planes
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(8 * w, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.norm(x))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != 4 * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, 4 * planes, 1, stride, bias=False),
                nn.BatchNorm2d(4 * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNet50(nn.Module):
    """ImageNet-style ResNet-50 (stride on the 3x3 conv, "v1.5"). width = stem / first-stage bottleneck width
    (64 = standard); stages use w, 2w, 4w, 8w with 4x expansion, so parameters scale ~ w^2 like ResNet18."""

    def __init__(self, num_classes=10, width=64, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        w = width
        self.norm = Normalize(mean, std)
        self.conv1 = nn.Conv2d(3, w, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        layers, in_planes = [], w
        for planes, n, stride in [(w, 3, 1), (2 * w, 4, 2), (4 * w, 6, 2), (8 * w, 3, 2)]:
            blocks = []
            for i in range(n):
                blocks.append(Bottleneck(in_planes, planes, stride if i == 0 else 1))
                in_planes = 4 * planes
            layers.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.bn1(self.conv1(self.norm(x)))), 3, 2, 1)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


DEFAULT_WIDTH = {"resnet18": 64, "resnet50": 64, "mobilenetv2": 32}


def build_model(name, num_classes=10, width=None, dataset="cifar10"):
    """width=None -> the architecture's standard width; normalization uses the dataset's mean / std."""
    from src.data import spec
    d = spec(dataset)
    kw = {"num_classes": num_classes, "width": width or DEFAULT_WIDTH[name], "mean": d["mean"], "std": d["std"]}
    if name == "resnet18":
        return ResNet18(**kw)
    if name == "resnet50":
        return ResNet50(**kw)
    if name == "mobilenetv2":
        from src.models.mobilenetv2 import MobileNetV2
        return MobileNetV2(**kw)
    raise ValueError(f"unknown model: {name}")
