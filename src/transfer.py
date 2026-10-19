"""Part 4: patch transfer between models (RQ2), every source model x every target model.

The patches come from Part 3 (patches/<attack>_<model>.pt). Each source contributes its "effective" patch
(for quantized sources: white-box or FP32-surrogate, whichever Part 3 reported), so the diagonal of every
matrix reproduces the Part 3 number for that model and checks that nothing changed in between.

Universal: each of the 10 targeted patches per size is evaluated on the target model over the full test set at
the same seeded placements as Part 3 (seed + size). Reports accuracy with the patch and target hit rate.

LaVAN: each per-image patch is pasted at its own saved location. The transfer rate counts only images that both models get
right clean and whose patch fooled the source model:
  untargeted: share the target model misclassifies; targeted: share it sends to the attack's target class.
Occlusion control: a random-noise patch at the same locations, same images, so the part of the transfer rate
that comes from covering pixels is visible next to it.

Usage:  python -m src.transfer --config configs/phase4/transfer.yaml
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from src.attack import stack_dataset
from src.attacks.patch import apply_patch, eval_universal
from src.data import eval_subset_indices, get_loaders
from src.models.load import load_model
from src.utils import get_device, load_config, save_result


def _phase3(run):
    return json.loads(Path("results", f"{run}.json").read_text())["metrics"]


def _mean_std(vals):
    return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals)}


def universal_matrix(models, sources, test, seed, physical=False, limit_targets=None):
    """{source: {target_model: {side_s: {patched_acc, target_success} mean/std over the 10 target classes}}}"""
    prefix = "universal_physical_" if physical else "universal_"
    out = {}
    for src in sources:
        saved, p3 = torch.load(Path("patches", f"{prefix}{src}.pt")), _phase3(prefix + src)
        out[src] = {}
        for tgt, model in models.items():
            row = {}
            for size, kinds in saved.items():
                kind = p3[f"side_{size}"]["effective_from"]
                classes = range(len(kinds[kind]))[:limit_targets]
                rs = [eval_universal(model, *test, kinds[kind][t].to(test[0].device), t, seed + size, physical)
                      for t in classes]
                row[f"side_{size}"] = {"from_kind": kind, "per_target": rs,
                                       **{m: _mean_std([r[m] for r in rs]) for m in ("patched_acc", "target_success")}}
            out[src][tgt] = row
            print(f"{prefix}{src} -> {tgt}  " + "  ".join(
                f"{k}: acc {v['patched_acc']['mean']:.4f} hit {v['target_success']['mean']:.4f}"
                for k, v in row.items()), flush=True)
    return out


@torch.no_grad()
def _predict(model, x, bs=1000):
    return torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


@torch.no_grad()
def lavan_matrix(models, sources, x, y, seed):
    """{source: {target_model: {side_s_mode: rates}}}; see module docstring for the definitions."""
    clean = {t: _predict(m, x) == y for t, m in models.items()}
    out = {}
    for src in sources:
        saved, p3 = torch.load(Path("patches", f"lavan_{src}.pt")), _phase3("lavan_" + src)
        out[src] = {t: {} for t in models}
        for key in sorted({(s, m) for s, m, _ in saved}):
            size, mode = key
            kind = p3[f"side_{size}_{mode}"]["effective_from"]
            d = saved[(size, mode, kind)]
            patch, rows, cols = d["patch"].float().to(x.device), d["rows"].to(x.device), d["cols"].to(x.device)
            src_ok = d["success"].to(x.device)
            target = None if d["target"] is None else d["target"].to(x.device)
            gen = torch.Generator().manual_seed(seed * 1000 + size * 10 + (mode == "targeted"))
            noise = torch.rand(patch.shape, generator=gen).to(x.device)
            x_adv, x_noise = apply_patch(x, patch, rows, cols), apply_patch(x, noise, rows, cols)
            for tgt, model in models.items():
                pa, pn = _predict(model, x_adv), _predict(model, x_noise)
                fooled = (pa != y) if target is None else (pa == target)
                fooled_noise = (pn != y) if target is None else (pn == target)
                base = src_ok & clean[src] & clean[tgt]
                out[src][tgt][f"side_{size}_{mode}"] = {
                    "from_kind": kind,
                    "n": int(base.sum()),
                    "transfer_rate": fooled[base].float().mean().item(),
                    "noise_rate": fooled_noise[base].float().mean().item(),
                    # unconditional: over all images the target gets right, whatever happened on the source
                    "success_on_correct": fooled[clean[tgt]].float().mean().item(),
                }
            print(f"lavan_{src} side {size} {mode}  " + "  ".join(
                f"{t}: {v[f'side_{size}_{mode}']['transfer_rate']:.3f}" for t, v in out[src].items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="2 sources x all targets, 2 classes, 1k images; no save")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t, seed, device = cfg["transfer"], cfg["seed"], get_device()
    torch.backends.cudnn.benchmark = True

    mcfgs = {Path(p).stem: load_config(p) for p in t["models"]}
    models = {c["run_name"]: load_model(c, device).requires_grad_(False) for c in mcfgs.values()}
    sources = list(models)

    _, test_loader = get_loaders(cfg["data_root"], 128, 0)
    test = stack_dataset(test_loader.dataset, device)
    subset = stack_dataset(test_loader.dataset, device, eval_subset_indices())
    limit = None
    if args.dry_run:
        sources, test, limit = sources[:2], (test[0][:1000], test[1][:1000]), 2

    metrics = {"models": list(models), "lavan": lavan_matrix(models, sources, *subset, seed)}
    metrics["universal"] = universal_matrix(models, sources, test, seed, limit_targets=limit)
    phys = [m for m in t.get("physical_models", []) if m in models]
    if phys:
        metrics["universal_physical"] = universal_matrix(
            {m: models[m] for m in phys}, phys[:2] if args.dry_run else phys, test, seed, True, limit)
    if args.dry_run:
        return
    cfg["model"] = "resnet18"
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
