"""Shared training loop and final metrics, used by the baseline and every compression arm."""
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import clean_accuracy, latency_ms, model_size


def cross_entropy(model, x, y):
    return nn.functional.cross_entropy(model(x), y)


def train_one_epoch(model, loader, opt, scaler, device, loss_fn=cross_entropy, amp=True, augment=None):
    model.train()
    use_amp = amp and device.type == "cuda"
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if augment:
            x = augment(x)
        with torch.autocast(device.type, enabled=use_amp):
            loss = loss_fn(model, x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def fit(model, train_loader, test_loader, device, t, ckpt, loss_fn=cross_entropy, amp=True):
    """SGD + Nesterov, cosine LR over t["epochs"]; saves the state_dict at ckpt.

    t["select"]: "best" (default, seed 42) keeps the best-test-accuracy epoch; "last" keeps the final epoch, so the
    test set plays no part in training (seeds 43+). On seed 42 the two differ by <= 0.13 pp in every run.
    t["mask_aug"] = {sizes, k, img (default 32)}: PatchCleanser mask augmentation on every training batch (defended
    models, Part 5); the masked pixels take the model's own normalization mean (the dataset mean).
    """
    last = t.get("select", "best") == "last"
    augment = None
    if t.get("mask_aug"):
        from src.defense.patchcleanser import MaskAugment
        augment = MaskAugment(t["mask_aug"]["sizes"], t["mask_aug"]["k"], t["mask_aug"].get("img", 32), device,
                              model.norm.mean.flatten().tolist())
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                          weight_decay=t["weight_decay"], nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    ckpt = Path(ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    resume = ckpt.with_suffix(".resume.pt")
    best, start = -1.0, 0
    state = None
    if resume.exists():
        try:
            state = torch.load(resume, map_location=device, weights_only=False)
        except RuntimeError as e:  # truncated by a power cut before fsync existed; start over
            print(f"unreadable resume file {resume} ({e}); training from scratch", flush=True)
            resume.unlink()
    if state is not None:
        if state["epochs"] == t["epochs"]:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            set_rng_state(state["rng"])
            best, start = state["best"], state["epoch"]
            print(f"resumed from {resume} after epoch {start}  best {best:.4f}", flush=True)
    for epoch in range(start, t["epochs"]):
        loss = train_one_epoch(model, train_loader, opt, scaler, device, loss_fn, amp, augment)
        sched.step()
        acc = clean_accuracy(model, test_loader, device)
        if acc > best or last:
            best = max(best, acc)
            torch.save(model.state_dict(), ckpt)
        # Per-epoch resume point (the lab machine has shut down mid-run); written atomically. The fsync matters:
        # without it a power cut can leave the renamed file with unflushed (zeroed) contents.
        tmp = resume.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "rng": get_rng_state(), "epoch": epoch + 1,
                        "epochs": t["epochs"], "best": best}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, resume)
        print(f"epoch {epoch + 1}/{t['epochs']}  loss {loss:.4f}  test_acc {acc:.4f}  best {best:.4f}", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    resume.unlink(missing_ok=True)
    return best


def get_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([r.cpu() for r in s["cuda"]])


def final_metrics(model, test_loader, device, cfg=None):
    from src.data import spec
    n = spec((cfg or {}).get("dataset", "cifar10"))["img"]
    return {
        "clean_acc": clean_accuracy(model, test_loader, device),
        **model_size(model),
        "latency_ms_bs1": latency_ms(model, device, (1, 3, n, n)),
    }
