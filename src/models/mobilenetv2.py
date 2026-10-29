"""MobileNetV2 (Sandler et al. 2018), CIFAR variant as in kuangliu/pytorch-cifar: stride 1 in the stem and in the
second stage, so a 32 x 32 input ends at 4 x 4 like ResNet-18. Second architecture (J7): depthwise-separable,
the usual edge model, so quantization and pruning hit it differently from a plain ResNet.

width = stem channel count (32 = standard). Every layer, including the final 1280-channel conv, scales by width / 32
(rounded to a multiple of 8), so the dense-small / KD widths can be matched to a parameter ratio as for ResNet-18.
Layer names (conv<i> / bn<i>, shortcut = Sequential(conv, bn)) follow ResNet18 so BN folding works unchanged.
"""
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import CIFAR10_MEAN, CIFAR10_STD, Normalize

# (expansion, out channels, blocks, stride)
CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def _ch(c, mult):
    return max(8, int(c * mult + 4) // 8 * 8)


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width=32, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        mult = width / 32
        self.norm = Normalize(mean, std)
        in_planes = _ch(32, mult)
        self.conv1 = nn.Conv2d(3, in_planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        blocks = []
        for expansion, out, n, stride in CFG:
            for i in range(n):
                blocks.append(InvertedResidual(in_planes, _ch(out, mult), expansion, stride if i == 0 else 1))
                in_planes = _ch(out, mult)
        self.layers = nn.Sequential(*blocks)
        last = _ch(1280, mult)
        self.conv2 = nn.Conv2d(in_planes, last, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(last)
        self.fc = nn.Linear(last, num_classes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(self.norm(x))))
        out = F.relu6(self.bn2(self.conv2(self.layers(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
