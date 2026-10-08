"""Print the results tables (markdown) from results/*.json, for PROGRESS.md and the paper.

Part 2: one row per trained model. Part 3: one row per attacked model, universal + LaVAN combined;
attack numbers are the "effective" ones (worse of white-box and FP32-surrogate for quantized models).
Part 3 ablation: physical-EOT universal patch vs location-only, side 8.

Usage:  python -m src.report
"""
import json
from pathlib import Path

BASELINE_MB = 42.625282287597656  # dense FP32 ResNet-18 parameters, same rule as compressed_size()
ORDER = {"none": 0, "quant": 1, "prune": 2, "distill": 4}


def _arm_key(r):
    c = r["compression"] or {"method": "none", "ratio": 1.0}
    kind = "dense_small" if r["metrics"].get("model_run", r["run"]).startswith("dense_small") else c["method"]
    return (3 if kind == "dense_small" else ORDER.get(c["method"], 3)), c.get("ratio", 1.0)


def _pct(v):
    return f"{v * 100:.1f}" if v is not None else "-"


def _get(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d


def attack_table(records):
    by_model = {}
    for r in records:
        name = r["attack"]["name"] + ("_physical" if r["attack"].get("physical") else "")
        by_model.setdefault(r["metrics"]["model_run"], {})[name] = r
    print("| Model | U5 acc | U5 hit | U8 acc | U8 hit | Noise8 acc | L5 robust acc | L8 robust acc | L5 tgt hit | L8 tgt hit |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model, rs in sorted(by_model.items(), key=lambda kv: _arm_key(next(iter(kv[1].values())))):
        u, lv = _get(rs, "universal", "metrics"), _get(rs, "lavan", "metrics")
        cells = [_get(u, "side_5", "effective_patched_acc", "mean"), _get(u, "side_5", "effective_target_success", "mean"),
                 _get(u, "side_8", "effective_patched_acc", "mean"), _get(u, "side_8", "effective_target_success", "mean"),
                 _get(u, "side_8", "noise_patch_acc"),
                 _get(lv, "side_5_untargeted", "effective", "robust_acc"),
                 _get(lv, "side_8_untargeted", "effective", "robust_acc"),
                 _get(lv, "side_5_targeted", "effective", "success_on_correct"),
                 _get(lv, "side_8_targeted", "effective", "success_on_correct")]
        print(f"| {model} | " + " | ".join(_pct(v) for v in cells) + " |")


def physical_table(records):
    """Physical-EOT ablation (universal, side 8) next to the location-only attack on the same model."""
    location = {r["metrics"]["model_run"]: r for r in records
                if r["attack"]["name"] == "universal" and not r["attack"].get("physical")}
    print("| Model | Location-only U8 acc | Location-only U8 hit | Physical-EOT U8 acc | Physical-EOT U8 hit |")
    print("|---|---|---|---|---|")
    phys = [r for r in records if r["attack"].get("physical")]
    for r in sorted(phys, key=_arm_key):
        model = r["metrics"]["model_run"]
        loc = _get(location.get(model), "metrics", "side_8")
        cells = [_get(loc, "effective_patched_acc", "mean"), _get(loc, "effective_target_success", "mean"),
                 _get(r["metrics"], "side_8", "effective_patched_acc", "mean"),
                 _get(r["metrics"], "side_8", "effective_target_success", "mean")]
        print(f"| {model} | " + " | ".join(_pct(v) for v in cells) + " |")


def main():
    rows, attacks = [], []
    for p in sorted(Path("results").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("attack"):
            attacks.append(r)
            continue
        m, c = r["metrics"], r["compression"]
        size = m.get("size_mb_compressed", BASELINE_MB)
        kind = "dense_small" if r["run"].startswith("dense_small") else c["method"]
        rows.append((ORDER.get(c["method"], 3) if kind != "dense_small" else 3, c.get("ratio", 1.0), r["run"], kind,
                     c.get("ratio", 1.0), BASELINE_MB / size, m["clean_acc"], m["nonzero_params"], size,
                     m["latency_ms_bs1"]))
    rows.sort()
    print("| Run | Arm | Nominal | Storage ratio | Clean acc | Nonzero params | Size (MB) | Latency (ms/img) |")
    print("|---|---|---|---|---|---|---|---|")
    for *_, run, kind, nominal, ratio, acc, nz, size, lat in rows:
        print(f"| {run} | {kind} | {nominal:g}x | {ratio:.2f}x | {acc * 100:.2f}% | {nz:,} | {size:.2f} | {lat:.2f} |")
    if attacks:
        print()
        attack_table(attacks)
        if any(r["attack"].get("physical") for r in attacks):
            print()
            physical_table(attacks)


if __name__ == "__main__":
    main()
