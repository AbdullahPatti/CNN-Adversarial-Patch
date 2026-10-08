"""Quantization arm: FP16 (2x), INT8 post-training quantization (4x), INT4 quantization-aware training (8x).

Quantization is simulated ("fake-quant"): weights and activations are rounded to the integer grid
and dequantized in float. This gives the same outputs as integer inference up to accumulation
rounding, stays on the GPU, and lets patch attacks backprop (rounding uses a straight-through
estimator; Part 4 checks for gradient masking).

Scheme:
- BN is folded into the preceding conv first (as an integer deployment would).
- Weights: symmetric, per-output-channel, weight_bits.
- Activations: the input of every conv / linear, asymmetric per-tensor, act_bits, with ranges from
  an EMA of batch min / max (calibration batches for PTQ; tracked during training for QAT).
- Residual adds, ReLU and pooling stay in float.
- 16 bits means an IEEE half cast instead of an integer grid.

Usage:  python -m src.compress.quantize --config configs/phase2/quant_int8.yaml
"""
import argparse
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_loaders
from src.engine import final_metrics, fit
from src.eval.metrics import clean_accuracy, compressed_size
from src.models.resnet import build_model
from src.utils import get_device, load_config, save_result, set_seed


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class HalfSTE(torch.autograd.Function):
    """FP16 round-trip in the forward pass, FP32 identity in the backward pass.

    A plain x.half().float() also casts the incoming gradient to FP16, where small gradients underflow
    to 0 (12.75% of input-pixel gradients with a batch-mean loss): gradient masking, not robustness.
    """
    @staticmethod
    def forward(ctx, x):
        return x.half().float()

    @staticmethod
    def backward(ctx, g):
        return g


def fake_quant_weight(w, bits):
    if bits >= 32:
        return w
    if bits == 16:
        return HalfSTE.apply(w)
    qmax = 2 ** (bits - 1) - 1
    dims = tuple(range(1, w.dim()))
    scale = w.detach().abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / qmax
    return RoundSTE.apply(w / scale).clamp(-qmax, qmax) * scale


class ActQuant(nn.Module):
    def __init__(self, bits, momentum=0.1):
        super().__init__()
        self.bits, self.momentum = bits, momentum
        self.observe = False  # set by calibrate(); always on in train mode
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x):
        if self.bits >= 32:
            return x
        if self.bits == 16:
            return HalfSTE.apply(x)
        if self.training or self.observe:
            lo, hi = x.detach().min(), x.detach().max()
            if not self.initialized:
                self.lo.copy_(lo), self.hi.copy_(hi), self.initialized.fill_(True)
            else:
                self.lo.lerp_(lo, self.momentum), self.hi.lerp_(hi, self.momentum)
        levels = 2 ** self.bits - 1
        lo, hi = self.lo.clamp(max=0.0), self.hi.clamp(min=0.0)  # keep 0 exactly representable
        scale = ((hi - lo) / levels).clamp_min(1e-8)
        zp = torch.round(-lo / scale)
        q = (RoundSTE.apply(x / scale) + zp).clamp(0, levels)
        return (q - zp) * scale


class QuantConv2d(nn.Module):
    def __init__(self, conv, weight_bits, act_bits):
        super().__init__()
        self.conv, self.weight_bits = conv, weight_bits
        self.aq = ActQuant(act_bits).to(conv.weight.device)

    def forward(self, x):
        c = self.conv
        return F.conv2d(self.aq(x), fake_quant_weight(c.weight, self.weight_bits), c.bias,
                        c.stride, c.padding, c.dilation, c.groups)


class QuantLinear(nn.Module):
    def __init__(self, linear, weight_bits, act_bits):
        super().__init__()
        self.linear, self.weight_bits = linear, weight_bits
        self.aq = ActQuant(act_bits).to(linear.weight.device)

    def forward(self, x):
        return F.linear(self.aq(x), fake_quant_weight(self.linear.weight, self.weight_bits), self.linear.bias)


@torch.no_grad()
def _fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    folded = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                       conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.device)
    folded.weight.copy_(conv.weight * s.view(-1, 1, 1, 1))
    b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    folded.bias.copy_(bn.bias + (b0 - bn.running_mean) * s)
    return folded


def fold_bn(model):
    """Fold every conv->BN pair of the CIFAR ResNet-18 into a single conv with bias (eval-mode BN)."""
    model.conv1, model.bn1 = _fold(model.conv1, model.bn1), nn.Identity()
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for blk in layer:
            blk.conv1, blk.bn1 = _fold(blk.conv1, blk.bn1), nn.Identity()
            blk.conv2, blk.bn2 = _fold(blk.conv2, blk.bn2), nn.Identity()
            if len(blk.shortcut) == 2:
                blk.shortcut = nn.Sequential(_fold(blk.shortcut[0], blk.shortcut[1]))
    return model


def wrap_quant(module, weight_bits, act_bits):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d(child, weight_bits, act_bits))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child, weight_bits, act_bits))
        else:
            wrap_quant(child, weight_bits, act_bits)
    return module


def quantize_model(model, weight_bits, act_bits):
    """Baseline-structured model (eval-mode BN stats loaded) -> folded, fake-quantized model."""
    return wrap_quant(fold_bn(model.eval()), weight_bits, act_bits)


def float_shadow(model):
    """Copy of a fake-quantized model with quantization switched off: the FP32 surrogate whose gradients
    attack the quantized model (BPDA, Part 3). For PTQ this is the BN-folded baseline; for QAT, the
    float weights QAT learned.
    """
    shadow = copy.deepcopy(model)
    for m in shadow.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.weight_bits = 32
        elif isinstance(m, ActQuant):
            m.bits = 32
    return shadow


@torch.no_grad()
def calibrate(model, loader, device, batches):
    aqs = [m for m in model.modules() if isinstance(m, ActQuant)]
    model.eval()
    for a in aqs:
        a.observe = True
    for x, _ in itertools.islice(loader, batches):
        model(x.to(device))
    for a in aqs:
        a.observe = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    t, c = cfg["train"], cfg["compression"]

    train_loader, test_loader = get_loaders(cfg["data_root"], t["batch_size"], cfg["num_workers"])
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(cfg["base_checkpoint"], map_location=device))
    acc_fp32 = clean_accuracy(model, test_loader, device)
    fold_bn(model.eval())
    acc_folded = clean_accuracy(model, test_loader, device)
    wrap_quant(model, c["weight_bits"], c["act_bits"])
    print(f"fp32 {acc_fp32:.4f}  bn-folded {acc_folded:.4f}", flush=True)

    calibrate(model, train_loader, device, c.get("calib_batches", 20))
    acc_ptq = clean_accuracy(model, test_loader, device)
    print(f"PTQ W{c['weight_bits']}A{c['act_bits']}  acc {acc_ptq:.4f}", flush=True)

    if c["mode"] == "qat":
        # Fake-quant rounding is done in FP32; AMP would put it on a coarser FP16 grid.
        fit(model, train_loader, test_loader, device, t, cfg["checkpoint"], amp=False)
    else:
        torch.save(model.state_dict(), cfg["checkpoint"])

    metrics = {
        **final_metrics(model, test_loader, device),
        **compressed_size(model, weight_bits=c["weight_bits"]),
        "acc_bn_folded_fp32": acc_folded,
        "acc_ptq": acc_ptq,
    }
    print(metrics)
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
