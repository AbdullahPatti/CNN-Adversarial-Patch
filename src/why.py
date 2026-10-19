"""J8 (C5): which measured model properties (src.analysis) track vulnerability, certification and transfer?

Each (model, seed) is one point: 39 points (13 models x 3 seeds), 36 for transfer (no baseline -> baseline).
Spearman rho with a 95% cluster-bootstrap CI that resamples whole models (a model's 3 seeds stay together, since
they are not independent). Partial rho controls for clean accuracy (both ranks regressed on the clean-accuracy
rank, residuals correlated), which asks "is it more than accuracy?". Vanilla properties go with the attacks and
transfer, defended-zoo properties with certification.

Usage:  python -m src.why      (prints markdown; writes results/summary/why.json)
"""
import json
from pathlib import Path

import numpy as np

from src.stats import MODELS, SEEDS, collect

A = Path("results/analysis")
BOOT = 10_000

PROPS = {  # analysis key -> short name
    "clean_acc_subset": "clean acc", "margin_median": "logit margin", "grad_l2_mean": "grad L2",
    "grad_conc_5": "grad conc. 5", "grad_conc_8": "grad conc. 8",
    "occ_grey_5_mean": "grey-5 occ.", "occ_grey_8_mean": "grey-8 occ.", "occ_noise_8_mean": "noise-8 occ.",
    "occ_grey_5_allpos": "grey-5 all-pos", "occ_grey_8_allpos": "grey-8 all-pos", "occ_noise_8_worst": "noise-8 worst",
}
SIM = {"agree_test": "agreement", "cka_penult": "CKA", "grad_cos_mean": "grad cosine"}
TARGETS_VAN = ["U5 acc", "U8 acc", "U8 hit", "L5 robust", "L5 tgt hit"]
TARGETS_DEF = ["cert5 def", "cert8 def"]
TARGETS_TR = ["T U8 hit full->comp", "T U8 hit comp->full", "T L5 rate full->comp", "T L5 rate comp->full"]


def rank(v):
    v = np.asarray(v, float)
    order = v.argsort(kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for u in np.unique(v):  # average ties
        idx = v == u
        r[idx] = r[idx].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def partial_spearman(x, y, z):
    rx, ry, rz = rank(x), rank(y), rank(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def boot(fn, groups, *cols, seed=0):
    """Cluster bootstrap over models: groups[i] = model of point i."""
    g = np.asarray(groups)
    ids = np.unique(g)
    members = [np.flatnonzero(g == i) for i in ids]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])
        cs = [np.asarray(c)[idx] for c in cols]
        if any(np.ptp(c) == 0 for c in cs):
            continue
        vals.append(fn(*cs))
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def props(defended):
    """prop[key][(model, seed index)] from results/analysis."""
    out = {}
    for model in MODELS:
        for i, sfx in enumerate(SEEDS.values()):
            p = A / f"{model}{sfx}{'_def' if defended else ''}.json"
            if not p.exists():
                continue
            for k, v in json.loads(p.read_text()).items():
                if isinstance(v, (int, float)):
                    out.setdefault(k, {})[(model, i)] = v
    return out


def corr_table(title, targets, pr, values, keys, models):
    print(f"### {title}\n")
    print("Spearman rho [95% cluster-bootstrap CI]; second line = partial rho given clean accuracy\n")
    print("| Property | " + " | ".join(targets) + " |")
    print("|---" * (len(targets) + 1) + "|")
    res = {}
    for key, name in keys.items():
        cells = []
        for t in targets:
            pts = [(m, i) for m in models for i in range(len(SEEDS))
                   if (m, i) in pr.get(key, {}) and values[t].get(m, [None] * 3)[i] is not None
                   and (m, i) in pr["clean_acc_subset"]]
            if len(pts) < 6:
                cells.append("-")
                continue
            x = [pr[key][p] for p in pts]
            y = [values[t][p[0]][p[1]] for p in pts]
            z = [pr["clean_acc_subset"][p] for p in pts]
            grp = [MODELS.index(p[0]) for p in pts]
            rho, (lo, hi) = spearman(x, y), boot(spearman, grp, x, y)
            entry = {"rho": rho, "lo": lo, "hi": hi, "n": len(pts)}
            txt = f"{rho:+.2f} [{lo:+.2f}, {hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}"
            if key != "clean_acc_subset":
                pr_, (plo, phi) = partial_spearman(x, y, z), boot(partial_spearman, grp, x, y, z)
                entry.update({"partial": pr_, "plo": plo, "phi": phi})
                txt += f"<br>{pr_:+.2f} [{plo:+.2f}, {phi:+.2f}]{'*' if plo > 0 or phi < 0 else ''}"
            res.setdefault(key, {})[t] = entry
            cells.append(txt)
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    return res


def main():
    values, _ = collect()
    van, dfd = props(False), props(True)
    out = {
        "vanilla": corr_table("Vanilla zoo: properties vs empirical patch attacks", TARGETS_VAN, van, values, PROPS, MODELS),
        "defended": corr_table("Defended zoo: properties vs PatchCleanser certified accuracy", TARGETS_DEF, dfd, values,
                               PROPS, MODELS),
        "transfer": corr_table("Vanilla zoo: similarity to the baseline vs transfer", TARGETS_TR, van, values,
                               {**SIM, "clean_acc_subset": "clean acc"}, MODELS[1:]),
    }
    # family means of the properties, for the text and the paper table
    print("### Property means over seeds (vanilla | defended)\n")
    keys = list(PROPS) + list(SIM)
    print("| Model | " + " | ".join(PROPS.get(k, SIM.get(k)) for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    means = {}
    for zoo, pr in (("vanilla", van), ("defended", dfd)):
        for m in MODELS:
            row = []
            for k in keys:
                v = [pr.get(k, {}).get((m, i)) for i in range(len(SEEDS))]
                v = [a for a in v if a is not None]
                mv = float(np.mean(v)) if v else None
                means.setdefault(zoo, {}).setdefault(m, {})[k] = mv
                row.append("-" if mv is None else f"{mv:.3f}")
            print(f"| {m}{' (def)' if zoo == 'defended' else ''} | " + " | ".join(row) + " |")
    out["means"] = means
    Path("results", "summary").mkdir(exist_ok=True)
    Path("results", "summary", "why.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
