"""J3 statistics for the follow-up phases: second settings (J5 GTSRB, J6 ImageNette, J7 MobileNetV2) and J9.

Same rules as src.stats: mean +/- std over training seeds, paired differences with a 95% t-interval over seeds
(seeds are paired: the attack / mask seed is the same for every training seed), and a paired image bootstrap for
certification. ImageNette has one seed, so its differences carry only the image bootstrap (certification) or no
interval at all. Metric names match src.stats; "5" / "8" are the setting's small / large patch (ImageNette 35 / 56 px).
J8 properties (results/analysis) are correlated with the attacks per setting, Spearman over (model, seed) points,
with the same cluster bootstrap over models as src.why.

J9 (CIFAR-10 ResNet-18, 3 seeds): untargeted universal patch on all 13 models ("UU" metrics) and LaVAN with the
Carlini-Wagner margin loss on baseline + 4x arms ("LM" metrics), next to the CE / targeted numbers of src.stats.

Usage:  python -m src.settings_stats      (prints markdown; writes results/summary/settings.json)
"""
import json
from pathlib import Path

import numpy as np

from src.setting_configs import ARMS, SETTINGS
from src.stats import MODELS, SEEDS, _g, _load, bootstrap, collect, mean_std, paired
from src.why import boot, partial_spearman, spearman

NAMES = {"gtsrb": "GTSRB, ResNet-18", "mbv2": "CIFAR-10, MobileNetV2", "inette": "ImageNette 224 px, ResNet-50"}
COMPS = [(f"{a} - base", a, "baseline") for a in ARMS[1:]] + [
    ("prune_4x - dense_small_4x", "prune_4x", "dense_small_4x"), ("kd_4x - dense_small_4x", "kd_4x", "dense_small_4x")]


def _sides(name):
    return SETTINGS[name]["sizes"]


def scalar_metrics(name):
    s, l = _sides(name)
    eff = lambda side, k: (lambda m: _g(m, f"side_{side}", k, "mean"))
    return {
        "clean": ("", False, lambda m: m["clean_acc"]),
        "U5 acc": ("universal_", False, eff(s, "effective_patched_acc")),
        "U5 hit": ("universal_", False, eff(s, "effective_target_success")),
        "U8 acc": ("universal_", False, eff(l, "effective_patched_acc")),
        "U8 hit": ("universal_", False, eff(l, "effective_target_success")),
        "Noise8 acc": ("universal_", False, lambda m: _g(m, f"side_{l}", "noise_patch_acc")),
        "L5 robust": ("lavan_", False, lambda m: _g(m, f"side_{s}_untargeted", "effective", "robust_acc")),
        "L8 robust": ("lavan_", False, lambda m: _g(m, f"side_{l}_untargeted", "effective", "robust_acc")),
        "L5 tgt hit": ("lavan_", False, lambda m: _g(m, f"side_{s}_targeted", "effective", "success_on_correct")),
        "L8 tgt hit": ("lavan_", False, lambda m: _g(m, f"side_{l}_targeted", "effective", "success_on_correct")),
        "cert5 vanilla": ("pc_", False, lambda m: _g(m, f"side_{s}", "certified_acc")),
        "cert8 vanilla": ("pc_", False, lambda m: _g(m, f"side_{l}", "certified_acc")),
        "def clean": ("", True, lambda m: m["clean_acc"]),
        "cert5 def": ("pc_", True, lambda m: _g(m, f"side_{s}", "certified_acc")),
        "cert8 def": ("pc_", True, lambda m: _g(m, f"side_{l}", "certified_acc")),
        "def clean5 (PC)": ("pc_", True, lambda m: _g(m, f"side_{s}", "defended_clean_acc")),
    }


def run_name(prefix, name, arm, seed, defended):
    return f"{prefix}{name}_{arm}_s{seed}{'_def' if defended else ''}"


def collect_setting(name):
    """values[metric][arm] = [v per seed]; images[metric][arm] = array (seeds, images)."""
    seeds = SETTINGS[name]["seeds"]
    s, l = _sides(name)
    values, images = {}, {}
    for metric, (prefix, defended, get) in scalar_metrics(name).items():
        for arm in ARMS:
            ms = [_load(run_name(prefix, name, arm, sd, defended)) for sd in seeds]
            values.setdefault(metric, {})[arm] = [get(m) if m else None for m in ms]
            if metric.startswith("cert") and all(ms):
                side = s if "5" in metric else l
                images.setdefault(metric, {})[arm] = np.array([_g(m, f"side_{side}", "certified_per_image") for m in ms],
                                                              dtype=float)
    transfer = [_load(f"transfer_{name}_s{sd}") for sd in seeds]
    for label, key, get in (("U8 hit", "universal", lambda c: c[f"side_{l}"]["target_success"]["mean"]),
                            ("L5 rate", "lavan", lambda c: c[f"side_{s}_untargeted"]["transfer_rate"]),
                            ("L5 noise", "lavan", lambda c: c[f"side_{s}_untargeted"]["noise_rate"])):
        for direction in ("full->comp", "comp->full"):
            row = values.setdefault(f"T {label} {direction}", {})
            for arm in ARMS[1:]:
                vals = []
                for sd, m in zip(seeds, transfer):
                    b, c = f"{name}_baseline_s{sd}", f"{name}_{arm}_s{sd}"
                    src, dst = (b, c) if direction == "full->comp" else (c, b)
                    vals.append(get(m[key][src][dst]) if m else None)
                row[arm] = vals
    return values, images


def collect_j9():
    """CIFAR-10 ResNet-18: untargeted universal (all 13 models) and margin-loss LaVAN (baseline + 4x), 3 seeds."""
    get_u = lambda side: (lambda m: _g(m, f"side_{side}", "effective_patched_acc", "mean"))
    get_l = lambda side, mode, k: (lambda m: _g(m, f"side_{side}_{mode}", "effective", k))
    metrics = {"UU5 acc": ("universal_untargeted_", get_u(5)), "UU8 acc": ("universal_untargeted_", get_u(8)),
               "LM5 robust": ("lavan_margin_", get_l(5, "untargeted", "robust_acc")),
               "LM8 robust": ("lavan_margin_", get_l(8, "untargeted", "robust_acc")),
               "LM5 tgt hit": ("lavan_margin_", get_l(5, "targeted", "success_on_correct")),
               "LM8 tgt hit": ("lavan_margin_", get_l(8, "targeted", "success_on_correct"))}
    values = {}
    for metric, (prefix, get) in metrics.items():
        for model in MODELS:
            ms = [_load(f"{prefix}{model}{sfx}") for sfx in SEEDS.values()]
            if any(ms):
                values.setdefault(metric, {})[model] = [get(m) if m else None for m in ms]
    return values


PROPS = {"clean_acc_subset": "clean acc", "margin_median": "logit margin", "grad_l2_mean": "grad L2",
         "grad_conc_8": "grad conc. 8", "occ_noise_8_mean": "noise-8 occ.", "occ_grey_8_allpos": "grey-8 all-pos"}
TARGETS_VAN = ["U8 acc", "L5 robust"]
TARGETS_DEF = ["cert5 def", "cert8 def"]


def props_setting(name, defended):
    out = {}
    for arm in ARMS:
        for i, sd in enumerate(SETTINGS[name]["seeds"]):
            p = Path("results/analysis", f"{run_name('', name, arm, sd, defended)}.json")
            if p.exists():
                for k, v in json.loads(p.read_text()).items():
                    if isinstance(v, (int, float)):
                        out.setdefault(k, {})[(arm, i)] = v
    return out


def correlations(name, values):
    """Spearman rho (and partial rho given clean accuracy) per property and target; needs >= 6 points."""
    res = {}
    for zoo, targets in (("vanilla", TARGETS_VAN), ("defended", TARGETS_DEF)):
        pr = props_setting(name, zoo == "defended")
        for key in PROPS:
            for t in targets:
                pts = [(a, i) for a in ARMS for i in range(len(SETTINGS[name]["seeds"]))
                       if (a, i) in pr.get(key, {}) and (a, i) in pr.get("clean_acc_subset", {})
                       and values[t].get(a, [None] * 3)[i] is not None]
                if len(pts) < 6:
                    continue
                x = [pr[key][p] for p in pts]
                y = [values[t][p[0]][p[1]] for p in pts]
                z = [pr["clean_acc_subset"][p] for p in pts]
                grp = [ARMS.index(p[0]) for p in pts]
                rho, (lo, hi) = spearman(x, y), boot(spearman, grp, x, y)
                e = {"rho": rho, "lo": lo, "hi": hi, "n": len(pts)}
                if key != "clean_acc_subset":
                    e["partial"] = partial_spearman(x, y, z)
                    e["plo"], e["phi"] = boot(partial_spearman, grp, x, y, z)
                res.setdefault(zoo, {}).setdefault(key, {})[t] = e
    return res


def _pct(m, s):
    return "-" if m is None else f"{m * 100:.1f} +/- {s * 100:.1f}"


def _diff(p):
    if p is None:
        return "-"
    if "lo" not in p:
        return f"{p['diff'] * 100:+.1f}"
    star = "*" if p["lo"] > 0 or p["hi"] < 0 else ""
    return f"{p['diff'] * 100:+.1f} [{p['lo'] * 100:+.1f}, {p['hi'] * 100:+.1f}]{star}"


def diff(va, vb):
    """Paired over seeds when there are >= 2; a single seed gives the point difference only."""
    p = paired(va, vb)
    if p is None:
        d = [x - y for x, y in zip(va or [], vb or []) if x is not None and y is not None]
        p = {"diff": d[0], "n": 1} if d else None
    return p


def table(rows_label, metrics, cell):
    print("| | " + " | ".join(metrics) + " |")
    print("|---" * (len(metrics) + 1) + "|")
    for r in rows_label:
        print(f"| {r} | " + " | ".join(cell(r, m) for m in metrics) + " |")
    print()


def main():
    out = {}
    groups = [["clean", "U5 acc", "U8 acc", "U8 hit", "Noise8 acc"],
              ["L5 robust", "L8 robust", "L5 tgt hit", "L8 tgt hit"],
              ["cert5 vanilla", "cert8 vanilla", "def clean", "cert5 def", "cert8 def"],
              ["T U8 hit full->comp", "T U8 hit comp->full", "T L5 rate full->comp", "T L5 rate comp->full",
               "T L5 noise full->comp"]]
    for name in SETTINGS:
        values, images = collect_setting(name)
        o = out[name] = {"means": {}, "diffs": {}, "boot": {}, "values": values}
        print(f"## {NAMES[name]} ({len(SETTINGS[name]['seeds'])} seed(s), patch sides {_sides(name)})\n")
        for metrics in groups:
            for metric in metrics:
                for arm in ARMS:
                    m, s, k = mean_std(values[metric].get(arm, []))
                    o["means"].setdefault(metric, {})[arm] = {"mean": m, "std": s, "n": k}
            table(ARMS, metrics, lambda a, m: _pct(o["means"][m][a]["mean"], o["means"][m][a]["std"]))
            for metric in metrics:
                for label, a, b in COMPS:
                    va, vb = values[metric].get(a), values[metric].get(b)
                    o["diffs"].setdefault(metric, {})[label] = diff(va, vb) if va and vb else None
            table([c[0] for c in COMPS], metrics, lambda lab, m: _diff(o["diffs"][m][lab]))
        print("Certification, paired image bootstrap of the seed-averaged difference (pp)\n")
        for metric in images:
            for label, a, b in COMPS:
                ia, ib = images[metric].get(a), images[metric].get(b)
                o["boot"].setdefault(metric, {})[label] = bootstrap(ia, ib) if ia is not None and ib is not None else None
        table([c[0] for c in COMPS], list(images), lambda lab, m: _diff(o["boot"][m][lab]))
        print("Defended - vanilla certified accuracy (pp)\n")
        for arm in ARMS:
            o["diffs"].setdefault("def - vanilla", {})[arm] = [
                diff(values[f"cert{s} def"][arm], values[f"cert{s} vanilla"][arm]) for s in (5, 8)]
        table(ARMS, ["cert5", "cert8"], lambda a, m: _diff(o["diffs"]["def - vanilla"][a][0 if m == "cert5" else 1]))
        corr = o["corr"] = correlations(name, values)
        if corr:
            print("J8 properties vs attacks / certification: Spearman rho [95% CI] | partial given clean acc\n")
            cols = [(z, t) for z in ("vanilla", "defended") for t in (TARGETS_VAN if z == "vanilla" else TARGETS_DEF)]
            print("| Property | " + " | ".join(f"{t} ({z[:3]})" for z, t in cols) + " |")
            print("|---" * (len(cols) + 1) + "|")
            for key, nice in PROPS.items():
                cells = []
                for z, t in cols:
                    e = corr.get(z, {}).get(key, {}).get(t)
                    if not e:
                        cells.append("-")
                        continue
                    c = f"{e['rho']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]"
                    if "partial" in e:
                        c += f" / {e['partial']:+.2f} [{e['plo']:+.2f}, {e['phi']:+.2f}]"
                    cells.append(c)
                print(f"| {nice} | " + " | ".join(cells) + " |")
            print()

    # J9 next to the main attack numbers of src.stats
    j9 = collect_j9()
    main_vals, _ = collect()
    for k in ("U8 acc", "U5 acc", "L5 robust", "L8 robust", "L5 tgt hit", "L8 tgt hit"):
        j9[k] = main_vals[k]
    o = out["j9"] = {"means": {}, "diffs": {}, "values": j9}
    print("## J9: attack completeness, CIFAR-10 ResNet-18 (3 seeds)\n")
    metrics = ["U5 acc", "UU5 acc", "U8 acc", "UU8 acc", "L5 robust", "LM5 robust", "L8 robust", "LM8 robust",
               "L5 tgt hit", "LM5 tgt hit"]
    for metric in metrics:
        for model in MODELS:
            m, s, k = mean_std(j9[metric].get(model, []))
            o["means"].setdefault(metric, {})[model] = {"mean": m, "std": s, "n": k}
    table(MODELS, metrics, lambda a, m: _pct(o["means"][m][a]["mean"], o["means"][m][a]["std"]))
    comps = [(f"{m} - base", m, MODELS[0]) for m in MODELS[1:]] + [
        (f"prune_{r} - dense_small_{r}", f"prune_{r}", f"dense_small_{r}") for r in ("2x", "4x", "8x")]
    for metric in metrics:
        for label, a, b in comps:
            va, vb = j9[metric].get(a), j9[metric].get(b)
            o["diffs"].setdefault(metric, {})[label] = paired(va, vb) if va and vb else None
    table([c[0] for c in comps], metrics, lambda lab, m: _diff(o["diffs"][m][lab]))
    # do the two attack variants rank the models the same way? Spearman over the seed means
    o["rank_agreement"] = {}
    for a, b in (("U8 acc", "UU8 acc"), ("U5 acc", "UU5 acc"), ("L5 robust", "LM5 robust"), ("L5 tgt hit", "LM5 tgt hit")):
        ms_ = [m for m in MODELS if o["means"][a][m]["mean"] is not None and o["means"][b][m]["mean"] is not None]
        rho = spearman([o["means"][a][m]["mean"] for m in ms_], [o["means"][b][m]["mean"] for m in ms_])
        o["rank_agreement"][f"{a}|{b}"] = {"rho": rho, "n": len(ms_)}
        print(f"rank agreement {a} vs {b}: Spearman {rho:+.2f} over {len(ms_)} models")
    Path("results", "summary").mkdir(exist_ok=True)
    Path("results", "summary", "settings.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
