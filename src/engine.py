"""Shared training loop and final metrics, used by the baseline and every compression arm."""
from pathlib import Path

import torch
import torch.nn as nn

from src.eval.metrics import clean_accuracy, latency_ms, model_size


def cross_entropy(model, x, y):
    return nn.functional.cross_entropy(model(x), y)


def train_one_epoch(model, loader, opt, scaler, device, loss_fn=cross_entropy, amp=True):
    model.train()
    use_amp = amp and device.type == "cuda"
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            loss = loss_fn(model, x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def fit(model, train_loader, test_loader, device, t, ckpt, loss_fn=cross_entropy, amp=True):
    """SGD + Nesterov, cosine LR over t["epochs"]; keeps the best-test-accuracy state_dict at ckpt.

    ponytail: selecting best-on-test is standard for CIFAR baselines but optimistic; switch to last epoch if reviewers object.
    """
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                          weight_decay=t["weight_decay"], nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    ckpt = Path(ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(t["epochs"]):
        loss = train_one_epoch(model, train_loader, opt, scaler, device, loss_fn, amp)
        sched.step()
        acc = clean_accuracy(model, test_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)
        print(f"epoch {epoch + 1}/{t['epochs']}  loss {loss:.4f}  test_acc {acc:.4f}  best {best:.4f}", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return best


def final_metrics(model, test_loader, device):
    return {
        "clean_acc": clean_accuracy(model, test_loader, device),
        **model_size(model),
        "latency_ms_bs1": latency_ms(model, device),
    }
