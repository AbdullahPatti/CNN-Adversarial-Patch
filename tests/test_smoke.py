"""CPU smoke check, no dataset download. Run: python -m tests.test_smoke"""
import torch

from src.attacks.patch import apply_patch, physical_transform
from src.compress.quantize import HalfSTE
from src.data import EVAL_SUBSET_SIZE, eval_subset_indices
from src.eval.metrics import latency_ms, model_size
from src.models.resnet import build_model


def main():
    model = build_model("resnet18").eval()
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 10)

    size = model_size(model)
    assert 11_000_000 < size["params"] < 11_300_000, size  # CIFAR ResNet-18 is ~11.17M
    # BatchNorm biases are initialized to exactly 0; a few random weights may also round to 0.
    bn_bias = sum(m.bias.numel() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert 0 <= size["params"] - bn_bias - size["nonzero_params"] < 100, size

    idx = eval_subset_indices()
    assert len(idx) == len(set(idx)) == EVAL_SUBSET_SIZE and idx == eval_subset_indices()

    assert latency_ms(model, torch.device("cpu"), warmup=1, iters=2) > 0

    # patch lands exactly at (row, col), per image
    out = apply_patch(torch.zeros(2, 3, 32, 32), torch.ones(3, 5, 5), torch.tensor([0, 27]), torch.tensor([3, 10]))
    assert out[0, :, 0:5, 3:8].eq(1).all() and out[1, :, 27:32, 10:15].eq(1).all() and out.sum() == 2 * 75
    p, mask = physical_transform(torch.rand(3, 8, 8), 4, torch.Generator().manual_seed(0))
    assert p.shape == (4, 3, 8, 8) and 0 < mask.mean() < 1

    # FP16 cast passes gradients in FP32: 1e-8 underflows to 0 in half precision
    x = torch.ones(4, requires_grad=True)
    HalfSTE.apply(x).mul(1e-8).sum().backward()
    assert x.grad.eq(1e-8).all()
    print("smoke ok", size)


if __name__ == "__main__":
    main()
