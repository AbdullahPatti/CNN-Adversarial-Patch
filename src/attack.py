"""Part 3: one patch attack (universal or LaVAN) against one trained model, at every patch size.

Quantized targets are attacked twice: white-box through the straight-through estimator, and BPDA with
the FP32 surrogate's gradients (`attack.surrogate: true`). The reported ("effective") number comes from
whichever of the two is stronger overall, so quantization can't look robust just because its gradients
are poor. It is one attack, not a per-image union, so quantized models get the same attack budget
(locations, restarts) as every other model.

Patches (and LaVAN locations / targets) go to patches/<run_name>.pt for the transfer study (Part 4).

Usage:  python -m src.attack --config configs/phase3/universal_kd_4x.yaml
"""
import argparse
import statistics
from pathlib import Path

import torch

from src.attacks.patch import eval_universal, lavan, train_universal
from src.data import eval_subset_indices, get_loaders
from src.models.load import load_model
from src.utils import get_device, load_config, save_result


def stack_dataset(ds, device, indices=None):
    idx = indices if indices is not None else range(len(ds))
    xs, ys = zip(*(ds[i] for i in idx))
    return torch.stack(xs).to(device), torch.tensor(ys, device=device)


@torch.no_grad()
def accuracy(model, x, y, bs=1000):
    return sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs)) / len(x)


def _mean_std(vals):
    return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals)}


def run_universal(a, seed, models, train_loader, test, subset, device):
    model, physical = models["whitebox"], a.get("physical", False)
    out, saved = {}, {}
    for size in a["sizes"]:
        per_target, patches = [], {k: [] for k in models}
        for target in a["targets"]:
            row = {"target": target}
            for kind, grad_model in models.items():
                gen = torch.Generator().manual_seed(seed * 1000 + size * 10 + target)
                patch, curve = train_universal(
                    grad_model, train_loader, size, target, a["steps"], a["lr"], device, gen, physical,
                    monitor=lambda p: eval_universal(model, *subset, p, target, seed + 1, physical),
                    monitor_every=a["monitor_every"])
                r = eval_universal(model, *test, patch, target, seed + size, physical)
                if physical:
                    r["digital"] = eval_universal(model, *test, patch, target, seed + size)
                row[kind] = {**r, "curve": curve}
                patches[kind].append(patch.cpu())
            per_target.append(row)
            print(f"size {size} target {target}  " + "  ".join(
                f"{k}: acc {row[k]['patched_acc']:.4f} hit {row[k]['target_success']:.4f}" for k in models), flush=True)
        best = max(models, key=lambda k: statistics.mean(t[k]["target_success"] for t in per_target))
        for t in per_target:
            t["effective"] = {m: t[best][m] for m in ("patched_acc", "target_success")}
        noise = torch.rand(3, size, size, generator=torch.Generator().manual_seed(seed * 1000 + size)).to(device)
        out[f"side_{size}"] = {
            "area_frac": size * size / 1024,
            "noise_patch_acc": eval_universal(model, *test, noise, -1, seed + size)["patched_acc"],
            "effective_from": best,
            **{f"{k}_{m}": _mean_std([t[k][m] for t in per_target])
               for k in [*models, "effective"] for m in ("patched_acc", "target_success")},
            "per_target": per_target,
        }
        saved[size] = {k: torch.stack(v) for k, v in patches.items()}
    return out, saved


def run_lavan(a, seed, models, x, y, device):
    model = models["whitebox"]
    with torch.no_grad():
        clean_correct = torch.cat([model(x[i:i + a["batch"]]).argmax(1) == y[i:i + a["batch"]]
                                   for i in range(0, len(x), a["batch"])])
    out, saved = {}, {}
    for size in a["sizes"]:
        for mode in a["modes"]:
            g = torch.Generator().manual_seed(seed * 1000 + size * 10 + (mode == "targeted"))
            target = (y.cpu() + 1 + torch.randint(0, 9, (len(y),), generator=g)).remainder(10).to(device) \
                if mode == "targeted" else None
            res, success = {}, {}
            for kind, grad_model in models.items():
                gen = torch.Generator().manual_seed(seed * 1000 + size * 10 + (mode == "targeted"))
                parts = []
                for i in range(0, len(x), a["batch"]):
                    j = slice(i, i + a["batch"])
                    parts.append(lavan(grad_model, model, x[j], y[j], size, a["steps"], a["step_size"],
                                       a["restarts"], gen, None if target is None else target[j]))
                patch, rows, cols, ok = (torch.cat(t) for t in zip(*parts))
                success[kind] = ok
                saved[(size, mode, kind)] = {"patch": patch.half().cpu(), "rows": rows.cpu(), "cols": cols.cpu(),
                                             "success": ok.cpu(), "target": None if target is None else target.cpu()}
            best = max(success, key=lambda k: success[k][clean_correct].float().mean().item())
            for kind, ok in [*success.items(), ("effective", success[best])]:
                r = {"success_on_correct": ok[clean_correct].float().mean().item()}
                if mode == "untargeted":
                    r["robust_acc"] = (clean_correct & ~ok).float().mean().item()
                else:
                    r["patched_acc"] = (clean_correct & ~ok).float().mean().item()  # upper bound: success != correct
                res[kind] = r
            out[f"side_{size}_{mode}"] = {"area_frac": size * size / 1024, "effective_from": best, **res}
            print(f"size {size} {mode}  " + "  ".join(f"{k}: {v}" for k, v in res.items()), flush=True)
    return out, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = load_config(cfg["model_config"])
    cfg["model"], cfg["compression"] = mcfg["model"], mcfg.get("compression")
    a, seed, device = cfg["attack"], cfg["seed"], get_device()
    torch.backends.cudnn.benchmark = True

    model = load_model(mcfg, device).requires_grad_(False)
    models = {"whitebox": model}
    if a.get("surrogate"):
        from src.compress.quantize import float_shadow
        models["surrogate_fp32"] = float_shadow(model)

    train_loader, test_loader = get_loaders(cfg["data_root"], a.get("batch_size", 128), cfg["num_workers"])
    test_ds = test_loader.dataset
    subset = stack_dataset(test_ds, device, eval_subset_indices())
    metrics = {"model_run": mcfg["run_name"]}
    if a["name"] == "universal":
        test = stack_dataset(test_ds, device)
        metrics["clean_acc"] = accuracy(model, *test)
        res, saved = run_universal(a, seed, models, train_loader, test, subset, device)
    else:
        metrics["clean_acc_subset"] = accuracy(model, *subset)
        res, saved = run_lavan(a, seed, models, *subset, device)
    metrics.update(res)

    Path("patches").mkdir(exist_ok=True)
    torch.save(saved, Path("patches", f"{cfg['run_name']}.pt"))
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
