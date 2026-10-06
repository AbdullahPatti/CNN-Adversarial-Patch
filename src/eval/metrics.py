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
