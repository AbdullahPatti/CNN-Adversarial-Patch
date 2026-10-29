"""PatchCleanser double masking (Xiang et al., USENIX Security 2022) on [0, 1] images.

Mask set: k positions per axis, stride s = ceil((n - p + 1) / k), mask side m = p + s - 1 (paper, Sec. 4.1); the
last position is clipped to the image edge. Any p x p patch anywhere in the image then lies fully inside at least
one mask. Masked pixels are set to the dataset mean, i.e. zero after the model's normalization, as in the
authors' code.

All predictions come from one table P2[b, i, j] = prediction with masks i and j both applied (P2[b, i, i] is the
one-mask prediction). From it:
  certified (Thm. 1): every two-mask prediction equals the true label, so no patch of side <= p can change the
    double-masking output for this image.
  double-masking output (Alg. 1): if all one-mask predictions agree, output it; otherwise, for each one-mask
    prediction that disagrees with the majority, output it if all second-round masks (with that first mask)
    agree with it; otherwise output the majority.
"""
import math

import torch

from src.models.resnet import CIFAR10_MEAN


def mask_positions(n, p, k):
    s = math.ceil((n - p + 1) / k)
    m = p + s - 1
    pos = sorted({min(i * s, n - m) for i in range(k)})
    return pos, m


def mask_set(n, p, k, device=None):
    """Boolean masks (M, 1, n, n), True = masked; M = (positions per axis)^2."""
    pos, m = mask_positions(n, p, k)
    masks = torch.zeros(len(pos) ** 2, 1, n, n, dtype=torch.bool, device=device)
    for a, r in enumerate(pos):
        for b, c in enumerate(pos):
            masks[a * len(pos) + b, :, r:r + m, c:c + m] = True
    return masks


class MaskAugment:
    """Training augmentation for the defended models: each image gets 0, 1 or 2 masks (equally likely) drawn
    from the union of the mask sets for every certified patch size, so the classifier learns to predict
    through the exact occlusions double masking produces."""

    def __init__(self, sizes, k, n=32, device=None, fill=CIFAR10_MEAN):  # fill: the dataset mean
        self.pool = torch.cat([mask_set(n, p, k, device) for p in sizes])
        self.fill = torch.tensor(fill, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        b, dev = x.size(0), x.device
        pick = torch.randint(0, len(self.pool), (2, b), device=dev)
        count = torch.randint(0, 3, (b,), device=dev)
        m = (self.pool[pick[0]] & (count >= 1).view(-1, 1, 1, 1)) | (self.pool[pick[1]] & (count == 2).view(-1, 1, 1, 1))
        return torch.where(m, self.fill.to(x.dtype), x)


@torch.no_grad()
def two_mask_preds(model, x, masks, fill=CIFAR10_MEAN, bs=1000):
    """P2 (B, M, M) long: prediction with masks i and j applied (symmetric; diagonal = one mask)."""
    fill = torch.tensor(fill, device=x.device).view(1, 3, 1, 1)
    nm = len(masks)
    p2 = torch.empty(len(x), nm, nm, dtype=torch.long, device=x.device)
    for i in range(nm):
        for j in range(i, nm):
            union = masks[i] | masks[j]
            pred = torch.cat([model(torch.where(union, fill, x[s:s + bs])).argmax(1) for s in range(0, len(x), bs)])
            p2[:, i, j] = p2[:, j, i] = pred
    return p2


def certified(p2, y):
    return (p2 == y.view(-1, 1, 1)).flatten(1).all(1)


def double_masking(p2):
    """Alg. 1 output (B,) from the two-mask table."""
    p1 = p2.diagonal(dim1=1, dim2=2)  # (B, M)
    majority = p1.mode(1).values
    out = majority.clone()
    for b in range(len(p2)):
        dis = (p1[b] != majority[b]).nonzero().squeeze(1)
        for i in dis.tolist():
            if (p2[b, i] == p1[b, i]).all():
                out[b] = p1[b, i]
                break
    return out
