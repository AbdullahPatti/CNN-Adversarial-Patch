"""J3: multi-seed statistics over results/*.json (seeds 42, 43, 44), markdown for PROGRESS.md and the paper.

For every metric and model: mean +/- std over the 3 training seeds. For every comparison (model vs baseline,
pruned vs dense-small at the same ratio, defended vs vanilla): the per-seed paired difference, its mean and a
95% t-interval over seeds (n = 3, df = 2). Seeds are paired because the attack / mask seed is the same for every
training seed. Where per-image flags exist (certification), a paired bootstrap over the fixed 1,000-image subset
of the seed-averaged per-image difference is reported as well; it covers image sampling, not training noise.
A difference is marked * only when the seed-level t-interval excludes 0 (the conservative one of the two).

Usage:  python -m src.stats            (also writes results/summary/stats.json; a subfolder so src.report ignores it)
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

SEEDS = {42: "", 43: "_s43", 44: "_s44"}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # two-sided 95% t quantiles, df = n - 1
MODELS = ["baseline_resnet18", "quant_2x_fp16", "quant_4x_int8", "quant_8x_int4", "prune_2x", "prune_4x", "prune_8x",
          "dense_small_2x", "dense_small_4x", "dense_small_8x", "kd_2x", "kd_4x", "kd_8x"]
BASE = MODELS[0]
BOOT = 10_000


def _load(name):
    p = Path("results", f"{name}.json")
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def _g(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d


# metric name -> (result file prefix, defended zoo?, getter on metrics); per-image getter for certification
SCALAR = {
    "clean": ("", False, lambda m: m["clean_acc"]),
    "U5 acc": ("universal_", False, lambda m: _g(m, "side_5", "effective_patched_acc", "mean")),
    "U5 hit": ("universal_", False, lambda m: _g(m, "side_5", "effective_target_success", "mean")),
    "U8 acc": ("universal_", False, lambda m: _g(m, "side_8", "effective_patched_acc", "mean")),
    "U8 hit": ("universal_", False, lambda m: _g(m, "side_8", "effective_target_success", "mean")),
    "Noise8 acc": ("universal_", False, lambda m: _g(m, "side_8", "noise_patch_acc")),
    "L5 robust": ("lavan_", False, lambda m: _g(m, "side_5_untargeted", "effective", "robust_acc")),
    "L8 robust": ("lavan_", False, lambda m: _g(m, "side_8_untargeted", "effective", "robust_acc")),
    "L5 tgt hit": ("lavan_", False, lambda m: _g(m, "side_5_targeted", "effective", "success_on_correct")),
    "L8 tgt hit": ("lavan_", False, lambda m: _g(m, "side_8_targeted", "effective", "success_on_correct")),
    "cert5 vanilla": ("pc_", False, lambda m: _g(m, "side_5", "certified_acc")),
    "cert8 vanilla": ("pc_", False, lambda m: _g(m, "side_8", "certified_acc")),
    "def clean": ("", True, lambda m: m["clean_acc"]),
    "cert5 def": ("pc_", True, lambda m: _g(m, "side_5", "certified_acc")),
    "cert8 def": ("pc_", True, lambda m: _g(m, "side_8", "certified_acc")),
    "def clean5 (PC)": ("pc_", True, lambda m: _g(m, "side_5", "defended_clean_acc")),
}
PER_IMAGE = {
    "cert5 vanilla": lambda m: _g(m, "side_5", "certified_per_image"),
    "cert8 vanilla": lambda m: _g(m, "side_8", "certified_per_image"),
    "cert5 def": lambda m: _g(m, "side_5", "certified_per_image"),
    "cert8 def": lambda m: _g(m, "side_8", "certified_per_image"),
}
TRANSFER = {  # (matrix, cell getter); source -> target
    "U8 hit": ("universal", lambda c: c["side_8"]["target_success"]["mean"]),
    "L5 rate": ("lavan", lambda c: c["side_5_untargeted"]["transfer_rate"]),
}


def _run(prefix, model, suffix, defended):
    return f"{prefix}{model}{suffix}{'_def' if defended else ''}"


def collect():
    """values[metric][model] = [v_seed42, v_seed43, v_seed44]; images[metric][model] = array (seeds, 1000)."""
    values, images = {}, {}
    for metric, (prefix, defended, get) in SCALAR.items():
        for model in MODELS:
            ms = [_load(_run(prefix, model, sfx, defended)) for sfx in SEEDS.values()]
            values.setdefault(metric, {})[model] = [get(m) if m else None for m in ms]
            if metric in PER_IMAGE and all(ms):
                images.setdefault(metric, {})[model] = np.array([PER_IMAGE[metric](m) for m in ms], dtype=float)
    for name, (key, get) in TRANSFER.items():
        tr = [_load(f"transfer{sfx}") for sfx in SEEDS.values()]
        for direction in ("full->comp", "comp->full"):
            row = values.setdefault(f"T {name} {direction}", {})
            for model in MODELS[1:]:
                vals = []
                for sfx, m in zip(SEEDS.values(), tr):
                    b, c = BASE + sfx, model + sfx
                    src, dst = (b, c) if direction == "full->comp" else (c, b)
                    vals.append(get(m[key][src][dst]) if m else None)
                row[model] = vals
    return values, images


def mean_std(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def paired(a, b):
    """Per-seed differences a - b: mean, 95% t-interval over seeds, n."""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 2:
        return None
    m, s = statistics.mean(d), statistics.stdev(d)
    h = T95[len(d) - 1] * s / math.sqrt(len(d))
    return {"diff": m, "lo": m - h, "hi": m + h, "n": len(d), "per_seed": d}


def bootstrap(a, b, seed=0):
    """Paired bootstrap over images of the seed-averaged per-image difference (a, b: arrays (seeds, images))."""
    d = (a - b).mean(0)
    idx = np.random.default_rng(seed).integers(0, len(d), (BOOT, len(d)))
    boots = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}


def comparisons():
    """(label, model a, model b): vs baseline for every arm, then RQ4 pruned vs dense-small at matched ratio."""
    out = [(f"{m} - base", m, BASE) for m in MODELS[1:]]
    out += [(f"prune_{r} - dense_small_{r}", f"prune_{r}", f"dense_small_{r}") for r in ("2x", "4x", "8x")]
    out += [(f"kd_{r} - dense_small_{r}", f"kd_{r}", f"dense_small_{r}") for r in ("2x", "4x", "8x")]
    return out


def _pct(v):
    return f"{v * 100:.1f}" if v is not None else "-"


def _diff(p):
    if p is None:
        return "-"
    star = "*" if p["lo"] > 0 or p["hi"] < 0 else ""
    return f"{p['diff'] * 100:+.1f} [{p['lo'] * 100:+.1f}, {p['hi'] * 100:+.1f}]{star}"


def main():
    values, images = collect()
    stats = {"means": {}, "diffs": {}, "boot": {}}
    groups = [("Clean accuracy and universal patches (effective attack)", ["clean", "U5 acc", "U5 hit", "U8 acc", "U8 hit",
                                                                          "Noise8 acc"]),
              ("LaVAN (effective attack)", ["L5 robust", "L8 robust", "L5 tgt hit", "L8 tgt hit"]),
              ("PatchCleanser certified accuracy", ["cert5 vanilla", "cert8 vanilla", "def clean", "cert5 def", "cert8 def"]),
              ("Transfer (source -> target)", [f"T {n} {d}" for n in TRANSFER for d in ("full->comp", "comp->full")])]
    for title, metrics in groups:
        print(f"### {title}: mean +/- std over seeds (%)\n")
        print("| Model | n | " + " | ".join(metrics) + " |")
        print("|---" * (len(metrics) + 2) + "|")
        for model in MODELS:
            cells, n = [], 0
            for metric in metrics:
                m, s, k = mean_std(values[metric].get(model, []))
                n = max(n, k)
                stats["means"].setdefault(metric, {})[model] = {"mean": m, "std": s, "n": k}
                cells.append("-" if m is None else f"{m * 100:.1f} +/- {s * 100:.1f}")
            print(f"| {model} | {n} | " + " | ".join(cells) + " |")
        print()
        print(f"### {title}: paired difference over seeds, mean [95% t-CI] (pp); * = CI excludes 0\n")
        print("| Comparison | " + " | ".join(metrics) + " |")
        print("|---" * (len(metrics) + 1) + "|")
        for label, a, b in comparisons():
            cells = []
            for metric in metrics:
                va, vb = values[metric].get(a), values[metric].get(b)
                p = paired(va, vb) if va and vb else None
                stats["diffs"].setdefault(metric, {})[label] = p
                cells.append(_diff(p))
            if any(c != "-" for c in cells):
                print(f"| {label} | " + " | ".join(cells) + " |")
        print()
    print("### Certification: paired image bootstrap of the seed-averaged difference (pp, 95% CI)\n")
    metrics = list(images)
    print("| Comparison | " + " | ".join(metrics) + " |")
    print("|---" * (len(metrics) + 1) + "|")
    for label, a, b in comparisons():
        cells = []
        for metric in metrics:
            ia, ib = images[metric].get(a), images[metric].get(b)
            p = bootstrap(ia, ib) if ia is not None and ib is not None else None
            stats["boot"].setdefault(metric, {})[label] = p
            cells.append(_diff(p))
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    print("### Defended - vanilla certified accuracy (same arm, paired over seeds, pp)\n")
    print("| Model | cert5 | cert8 |")
    print("|---|---|---|")
    for model in MODELS:
        ps = [paired(values[f"cert{s} def"][model], values[f"cert{s} vanilla"][model]) for s in (5, 8)]
        stats["diffs"].setdefault("def - vanilla", {})[model] = ps
        print(f"| {model} | " + " | ".join(_diff(p) for p in ps) + " |")
    Path("results", "summary").mkdir(exist_ok=True)
    Path("results", "summary", "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
