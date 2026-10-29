"""Data for the project website: docs/data.js from paper/generated/macros.json, one CIFAR-10 test image and the
baseline's universal 8 px patches (one per target class), so no number on the site is typed by hand.

Usage:  python -m src.site_data      (after python -m src.paper_assets)
"""
import json
import pickle
import re
from pathlib import Path

import torch

FAMILIES = {"Quantization": ["quant_2x_fp16", "quant_4x_int8", "quant_8x_int4"],
            "Pruning": ["prune_2x", "prune_4x", "prune_8x"],
            "Dense-small": ["dense_small_2x", "dense_small_4x", "dense_small_8x"],
            "Distillation": ["kd_2x", "kd_4x", "kd_8x"]}
METRICS = ["clean", "defclean", "cert5def", "cert8def", "cert5vanilla", "u8acc", "l5robust", "noise8acc",
           "tu8hitfulltocomp", "tl5ratefulltocomp", "tl5ratecomptofull"]
SETTINGS = {"cifar10": "CIFAR-10 / ResNet-18", "gtsrb": "GTSRB / ResNet-18", "mbv2": "CIFAR-10 / MobileNetV2",
            "inette": "ImageNette 224px / ResNet-50"}
ARMS = {"quant_4x_int8": "INT8", "prune_4x": "Pruning 75%", "dense_small_4x": "Dense-small", "kd_4x": "Distillation"}
BASE = "baseline_resnet18"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def num(s):
    if s is None:
        return None
    s = re.sub(r"\$-\$", "-", str(s)).replace("$", "").replace("\\", "")
    try:
        return float(s)
    except ValueError:
        return None


def main():
    m = json.loads(Path("paper/generated/macros.json").read_text())
    models = {k: {t: num(m.get(f"{t}:{k}")) for t in METRICS} for k in [BASE] + sum(FAMILIES.values(), [])}
    settings = {}
    for s, label in SETTINGS.items():
        rows = {}
        for a, name in ARMS.items():
            if s == "cifar10":
                key = f"d:cert5def:{a}-{BASE}"
            elif s == "inette":
                key = f"inette:b:cert5def:{a}-baseline"
            else:
                key = f"{s}:d:cert5def:{a}-baseline"
            rows[name] = [num(m.get(key)), num(m.get(key + ":lo")), num(m.get(key + ":hi"))]
        settings[label] = rows
    with open("data/cifar-10-batches-py/test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    idx = 6  # an automobile
    img = d[b"data"][idx].reshape(3, 32, 32).transpose(1, 2, 0).tolist()
    patches = torch.load("patches/universal_baseline_resnet18.pt", map_location="cpu")[8]["whitebox"]
    patches = (patches.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).tolist()
    keys = {k: m[k] for k in ("nseeds", "keepworst5", "why:defended:occnoise8mean:cert5def",
                              "why:transfer:agreetest:tl5ratecomptofull", "d:cert5def:prune_8x-dense_small_8x",
                              "d:cert5def:dense_small_8x-baseline_resnet18:abs")}
    out = {"families": FAMILIES, "models": models, "settings": settings, "image": img,
           "imageLabel": CLASSES[d[b"labels"][idx]], "patches": patches, "classes": CLASSES,
           "keys": {k: num(v) for k, v in keys.items()}}
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data.js").write_text("window.DATA=" + json.dumps(out, separators=(",", ":")) + ";\n")
    print("docs/data.js written,", sum(v is None for r in models.values() for v in r.values()), "missing values")


if __name__ == "__main__":
    main()
