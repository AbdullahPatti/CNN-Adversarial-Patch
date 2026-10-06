"""CPU smoke check, no dataset download. Run: python -m tests.test_smoke"""
import torch

from src.data import EVAL_SUBSET_SIZE, eval_subset_indices
from src.eval.metrics import latency_ms, model_size
from src.models.resnet import build_model


def main():
    model = build_model("resnet18").eval()
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 10)

    size = model_size(model)
    assert 11_000_000 < size["params"] < 11_300_000, size  # CIFAR ResNet-18 is ~11.17M
    assert size["nonzero_params"] == size["params"]

    idx = eval_subset_indices()
    assert len(idx) == len(set(idx)) == EVAL_SUBSET_SIZE and idx == eval_subset_indices()

    assert latency_ms(model, torch.device("cpu"), warmup=1, iters=2) > 0
    print("smoke ok", size)


if __name__ == "__main__":
    main()
