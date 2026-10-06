"""ResNet-18, CIFAR variant (3x3 stem, no max-pool), as in He et al. 2016 / kuangliu/pytorch-cifar.

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
    def __init__(self, num_classes=10):
        super().__init__()
        self.norm = Normalize(CIFAR10_MEAN, CIFAR10_STD)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers, in_planes = [], 64
        for planes, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers.append(nn.Sequential(BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes, 1)))
            in_planes = planes
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.norm(x))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(name, num_classes=10):
    # ponytail: one arch for now; KD student and dense-small control get added here in Part 2/3.
    if name == "resnet18":
        return ResNet18(num_classes)
    raise ValueError(f"unknown model: {name}")
