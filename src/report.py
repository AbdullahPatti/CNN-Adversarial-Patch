"""Print the results table (markdown) from results/*.json, for PROGRESS.md and the paper.

Usage:  python -m src.report
"""
import json
from pathlib import Path

BASELINE_MB = 42.625282287597656  # dense FP32 ResNet-18 parameters, same rule as compressed_size()
ORDER = {"none": 0, "quant": 1, "prune": 2, "distill": 4}


def main():
    rows = []
    for p in sorted(Path("results").glob("*.json")):
        r = json.loads(p.read_text())
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


if __name__ == "__main__":
    main()
