"""Clean accuracy, model size, latency. Shared by every compression arm."""
import time

import torch


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def model_size(model):
    """Parameter counts and dense FP32 storage size.

    size_mb_fp32 is the dense on-disk size of the state_dict. Compressed sizes
    (sparse storage for pruning, bit-width for quantization) are computed by
    each compression arm, since they depend on the storage format.
    """
    params = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    size_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return {"params": params, "nonzero_params": nonzero, "size_mb_fp32": size_bytes / 2**20}


@torch.no_grad()
def latency_ms(model, device, input_shape=(1, 3, 32, 32), warmup=20, iters=200):
    """Mean wall-clock forward time per single image (batch size 1)."""
    model.eval()
    x = torch.rand(input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def compressed_size(model, weight_bits=32, sparse=False):
    """Shipped storage size, the size metric compared across compression arms.

    Conv / linear weights cost weight_bits each; every other parameter (biases, BN) costs 32 bits.
    sparse=True (pruning): only nonzero weights are stored, plus a 1-bit-per-weight mask.
    Buffers (BN running stats, normalization, activation ranges) are not counted.
    ratio_vs_baseline compares against the dense FP32 width-64 ResNet-18 under the same rule.
    """
    global _BASELINE_BITS
    if _BASELINE_BITS is None:
        from src.models.resnet import build_model
        _BASELINE_BITS = _storage_bits(build_model("resnet18"), 32, False)
    b = _storage_bits(model, weight_bits, sparse)
    return {"size_mb_compressed": b / 8 / 2**20, "ratio_vs_baseline": _BASELINE_BITS / b}


def _storage_bits(model, weight_bits, sparse):
    bits = 0
    weights = set()
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            w = mod.weight
            weights.add(id(w))
            if sparse:
                bits += int((w != 0).sum()) * weight_bits + w.numel()
            else:
                bits += w.numel() * weight_bits
    return bits + sum(p.numel() * 32 for p in model.parameters() if id(p) not in weights)


_BASELINE_BITS = None
