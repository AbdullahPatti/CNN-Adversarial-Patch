"""Global unstructured magnitude pruning + fine-tuning, starting from the baseline.

All conv and linear weights compete in one global magnitude ranking; BN and biases are not pruned.
Masks stay attached during fine-tuning so pruned weights remain exactly zero.

Usage:  python -m src.compress.prune --config configs/phase2/prune_75.yaml
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.data import get_loaders
from src.engine import final_metrics, fit
from src.eval.metrics import clean_accuracy, compressed_size
from src.models.resnet import build_model
from src.utils import get_device, load_config, save_result, set_seed


def prunable(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def weight_sparsity(model):
    ws = [m.weight for m, _ in prunable(model)]
    return sum(int((w == 0).sum()) for w in ws) / sum(w.numel() for w in ws)


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

    prune.global_unstructured(prunable(model), pruning_method=prune.L1Unstructured, amount=c["sparsity"])
    acc_before_ft = clean_accuracy(model, test_loader, device)
    print(f"sparsity {weight_sparsity(model):.4f}  acc after one-shot prune {acc_before_ft:.4f}", flush=True)

    fit(model, train_loader, test_loader, device, t, cfg["checkpoint"])

    # Make masks permanent so the checkpoint loads into a plain ResNet-18.
    for m, name in prunable(model):
        prune.remove(m, name)
    torch.save(model.state_dict(), cfg["checkpoint"])

    metrics = {
        **final_metrics(model, test_loader, device),
        **compressed_size(model, weight_bits=32, sparse=True),
        "weight_sparsity": weight_sparsity(model),
        "acc_after_prune_before_ft": acc_before_ft,
    }
    print(metrics)
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
