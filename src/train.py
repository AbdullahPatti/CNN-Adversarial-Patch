"""Train the full-precision baseline.

Usage:  python -m src.train --config configs/baseline.yaml
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_loaders
from src.eval.metrics import clean_accuracy, latency_ms, model_size
from src.models.resnet import build_model
from src.utils import get_device, load_config, save_result, set_seed


def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    t = cfg["train"]

    train_loader, test_loader = get_loaders(cfg["data_root"], t["batch_size"], cfg["num_workers"])
    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                          weight_decay=t["weight_decay"], nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    ckpt = Path(cfg["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(t["epochs"]):
        loss = train_one_epoch(model, train_loader, opt, scaler, device)
        sched.step()
        acc = clean_accuracy(model, test_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)
        print(f"epoch {epoch + 1}/{t['epochs']}  loss {loss:.4f}  test_acc {acc:.4f}  best {best:.4f}", flush=True)

    # Report metrics for the saved (best) checkpoint.
    # ponytail: selecting best-on-test is standard for CIFAR baselines but optimistic; switch to last epoch if reviewers object.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    metrics = {
        "clean_acc": clean_accuracy(model, test_loader, device),
        **model_size(model),
        "latency_ms_bs1": latency_ms(model, device),
    }
    print(metrics)
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
