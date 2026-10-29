"""J8 (C5): measurable model properties that may explain patch vulnerability, certification and transfer.

For every trained model (13 arms x 3 seeds, vanilla and defended zoo) on the fixed 1,000-image subset:
  - occlusion: accuracy with a grey (dataset-mean) or uniform-noise square of side 5 / 8 at every position of
    a stride-3 grid -> mean, worst position, and share of images correct at every position
  - logit margin (true-class logit minus the best other logit), mean and median
  - input gradient of the cross-entropy: L2 norm, and concentration = largest share of |grad| (summed over
    channels) inside one p x p window, p = 5 / 8 ("how patchable is the model")
  - similarity to the baseline of the same seed and zoo: top-1 agreement on the full 10k test set, linear CKA
    of the penultimate features, and cosine of the input gradients (gradient alignment; Demontis et al. 2019)
Quantized models are differentiated through their STE (the same gradients the white-box attack uses).
One JSON per model in results/analysis/ (a subfolder, so src.report / src.stats ignore it); finished models are
skipped, so the run is restartable.

Follow-up settings (J5-J7: gtsrb_*, mbv2_*, inette_*) use the same measures with their own dataset mean as the grey
fill and their own baseline. Keys keep the CIFAR names: "5" / "8" mean the small / large patch side of the setting
(ImageNette: 35 / 56 px, occlusion stride 21 = 3 x 7 so the grid has the same density, first 500 subset images as
in its attacks and certification).

Usage:  python -m src.analysis            (all models, ~1 GPU-h for CIFAR-10 ResNet-18, ~2 h for the follow-ups)
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.attack import stack_dataset
from src.data import eval_subset_indices, get_loaders, spec
from src.models.load import load_model
from src.utils import get_device, load_config

OUT = Path("results/analysis")
CONFIG_DIRS = ["configs/phase2", "configs/seed42_def/phase2", "configs/seed43/phase2", "configs/seed43_def/phase2",
               "configs/seed44/phase2", "configs/seed44_def/phase2", "configs/*/seed4?/phase2", "configs/*/seed4?_def/phase2"]
# dataset -> (small, large patch side, occlusion stride, images of the fixed subset)
GEOM = {"cifar10": (5, 8, 3, None), "gtsrb": (5, 8, 3, None), "imagenette": (35, 56, 21, 500)}


def model_configs():
    cfgs = {"baseline_resnet18": load_config("configs/baseline.yaml")}
    for d in CONFIG_DIRS:
        for p in sorted(Path().glob(f"{d}/*.yaml")):
            c = load_config(str(p))
            cfgs[c["run_name"]] = c
    return cfgs


def baseline_of(run):
    """Same seed, same zoo: prune_4x_s43_def -> baseline_resnet18_s43_def, mbv2_kd_4x_s42 -> mbv2_baseline_s42."""
    setting = run.split("_")[0]
    if setting in ("gtsrb", "mbv2", "inette"):
        tail = run[run.rindex("_s4"):]  # _s42 / _s43_def
        return f"{setting}_baseline{tail}"
    suffix = "".join(s for s in ("_s43", "_s44") if s in run) + ("_def" if run.endswith("_def") else "")
    return "baseline_resnet18" + suffix


@torch.no_grad()
def logits_feats(model, x, bs=None):
    bs = bs or (500 if x.size(-1) <= 32 else 50)
    feats = []
    h = model.fc.register_forward_hook(lambda m, i, o: feats.append(i[0].float()))
    try:
        logits = torch.cat([model(x[s:s + bs]).float() for s in range(0, len(x), bs)])
    finally:
        h.remove()
    return logits, torch.cat(feats)


def input_grad(model, x, y, bs=None):
    bs = bs or (250 if x.size(-1) <= 32 else 25)
    gs = []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs].clone().requires_grad_(True)
        loss = F.cross_entropy(model(xb).float(), y[s:s + bs], reduction="sum")
        gs.append(torch.autograd.grad(loss, xb)[0].detach())
    return torch.cat(gs)


@torch.no_grad()
def occlusion(model, x, y, p, kind, gen, mean, stride, bs=1000):
    """Correctness (positions, images) with a p x p occluder at every position of a stride grid."""
    n = x.size(-1)
    pos = list(range(0, n - p + 1, stride))
    if pos[-1] != n - p:
        pos.append(n - p)
    fill = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    ok = []
    for r in pos:
        for c in pos:
            xo = x.clone()
            if kind == "grey":
                xo[:, :, r:r + p, c:c + p] = fill
            else:
                xo[:, :, r:r + p, c:c + p] = torch.rand(len(x), 3, p, p, generator=gen, device="cpu").to(x.device)
            ok.append(torch.cat([model(xo[s:s + bs]).argmax(1) == y[s:s + bs] for s in range(0, len(x), bs)]))
    return torch.stack(ok)


@torch.no_grad()
def predict(model, xt):
    """Top-1 on the full test set; xt may live on the CPU (ImageNette at 224 px), batches go to the model's device."""
    dev, bs = next(model.parameters()).device, (1000 if xt.size(-1) <= 32 else 100)
    return torch.cat([model(xt[s:s + bs].to(dev)).argmax(1).cpu() for s in range(0, len(xt), bs)])


def linear_cka(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return ((a.T @ b).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm())).item()


def concentration(g, p):
    m = g.abs().sum(1, keepdim=True)
    win = F.avg_pool2d(m, p, stride=1) * p * p
    return (win.flatten(1).max(1).values / m.flatten(1).sum(1).clamp_min(1e-12))


def measure(model, x, y, xt, yt, ref, gen, dataset="cifar10"):
    logits, feats = logits_feats(model, x)
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = logits.scatter(1, y.view(-1, 1), float("-inf")).max(1).values
    margin = true - other
    g = input_grad(model, x, y)
    out = {"clean_acc_subset": (logits.argmax(1) == y).float().mean().item(),
           "margin_mean": margin.mean().item(), "margin_median": margin.median().item(),
           "grad_l2_mean": g.flatten(1).norm(dim=1).mean().item()}
    small, large, stride, _ = GEOM[dataset]
    bs = 1000 if x.size(-1) <= 32 else 100
    for key, p in ((5, small), (8, large)):
        out[f"grad_conc_{key}"] = concentration(g, p).mean().item()
        for kind in ("grey", "noise"):
            ok = occlusion(model, x, y, p, kind, gen, spec(dataset)["mean"], stride, bs).float()
            out[f"occ_{kind}_{key}_mean"] = ok.mean().item()
            out[f"occ_{kind}_{key}_worst"] = ok.mean(1).min().item()
            out[f"occ_{kind}_{key}_allpos"] = ok.min(0).values.mean().item()
    if ref is not None:
        with torch.no_grad():
            pt = predict(model, xt)
        out["agree_test"] = (pt == ref["pred_test"]).float().mean().item()
        out["cka_penult"] = linear_cka(feats, ref["feats"])
        cos = F.cosine_similarity(g.flatten(1), ref["grad"].flatten(1), dim=1)
        out["grad_cos_mean"] = cos.mean().item()
        out["grad_cos_median"] = cos.median().item()
    return out, {"feats": feats, "grad": g}


def main():
    device = get_device()
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = model_configs()
    runs = sys.argv[1:] or [r for r in cfgs]
    # baselines first, so every compressed model can be compared with its own baseline
    runs = sorted(runs, key=lambda r: (not r.startswith("baseline") and "_baseline_" not in r, r))
    by_ds = {}
    for r in runs:
        by_ds.setdefault(cfgs[r].get("dataset", "cifar10"), []).append(r)
    for ds, ds_runs in by_ds.items():
        todo = [r for r in ds_runs if not (OUT / f"{r}.json").exists()]
        if not todo:
            print(f"[skip] {ds}: all {len(ds_runs)} done", flush=True)
            continue
        n_img = GEOM[ds][3]
        _, test_loader = get_loaders("./data", 128, 0, ds)
        x, y = stack_dataset(test_loader.dataset, device, eval_subset_indices(ds)[:n_img])
        xt, yt = stack_dataset(test_loader.dataset, "cpu" if ds == "imagenette" else device)
        refs = {}
        for run in ds_runs:
            path = OUT / f"{run}.json"
            base = baseline_of(run)
            need_ref = run == base and any(baseline_of(r) == run and not (OUT / f"{r}.json").exists()
                                           for r in ds_runs if r != run)
            if path.exists() and not need_ref:
                print(f"[skip] {run}", flush=True)
                continue
            t0 = time.time()
            model = load_model(cfgs[run], device).requires_grad_(False)
            gen = torch.Generator().manual_seed(0)  # same noise occluders for every model
            ref = refs.get(base) if run != base else None
            out, keep = measure(model, x, y, xt, yt, ref, gen, ds)
            if run == base:
                keep["pred_test"] = predict(model, xt)
                refs[run] = keep
            out = {"run": run, "baseline": base, **out}
            if run == base and path.exists():  # recomputed only to serve as reference; keep the stored numbers
                out = json.loads(path.read_text())
            path.write_text(json.dumps(out, indent=1))
            print(f"[done] {run} {time.time() - t0:.0f}s  " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()
                                                                         if isinstance(v, float)), flush=True)
            del model
            torch.cuda.empty_cache()
        del x, y, xt, yt, refs


if __name__ == "__main__":
    main()
