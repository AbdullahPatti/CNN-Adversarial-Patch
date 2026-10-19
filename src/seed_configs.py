"""Clone the seed-42 pipeline for another training seed and/or as the defended (mask-augmented) model zoo.

Vanilla, seed S != 42: configs/seed<S>/{phase2,phase3,phase4,phase5}: the 13 models, Part 3 attacks, Part 4
transfer and Part 5 certification. Run names, checkpoints and patches get a _s<S> suffix.
Defended (--defended), any seed: configs/seed<S>_def/{phase2,phase5}: the 13 models trained with PatchCleanser mask
augmentation in every training / fine-tuning / QAT / KD run (the KD teacher and the pruning / quantization base are
the defended baseline of the same seed), then certified. Suffix _def (seed 42) or _s<S>_def.

Trained runs keep the final epoch (train.select = last). The attack seed is unchanged, so every training seed
faces the same patch placements, LaVAN targets and noise controls (paired comparison across seeds).
The physical-EOT ablation is not cloned (single-seed ablation).

Usage:  python -m src.seed_configs 43 44
        python -m src.seed_configs --defended 42 43 44
"""
import argparse
from pathlib import Path

import yaml

from src.utils import load_config

MASK_AUG = {"sizes": [5, 8], "k": 6}


def _suffix(path, tag):
    p = Path(path)
    return p.with_name(f"{p.stem}{tag}{p.suffix}").as_posix()


def _dump(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def clone(s, defended=False):
    tag = ("" if s == 42 else f"_s{s}") + ("_def" if defended else "")
    assert tag, "seed 42 vanilla is the original pipeline"
    out = Path("configs", f"seed{s}" + ("_def" if defended else ""))
    model_cfgs = {"configs/baseline.yaml": out / "phase2" / "00_baseline.yaml",
                  **{p.as_posix(): out / "phase2" / p.name for p in sorted(Path("configs/phase2").glob("*.yaml"))}}
    for src, dst in model_cfgs.items():
        c = load_config(src)
        c["run_name"] += tag
        c["seed"] = s
        c["checkpoint"] = _suffix(c["checkpoint"], tag)
        if "base_checkpoint" in c:
            c["base_checkpoint"] = _suffix(c["base_checkpoint"], tag)
        if "train" in c:
            c["train"]["select"] = "last"
            if defended:
                c["train"]["mask_aug"] = MASK_AUG
        _dump(c, dst)
    for p in sorted(Path("configs/phase5").glob("*.yaml")):
        c = load_config(p)
        c["run_name"] += tag
        c["model_config"] = model_cfgs[c["model_config"]].as_posix()
        c["defense"]["fine_tuned"] = defended
        _dump(c, out / "phase5" / p.name)
    if not defended:
        for p in sorted(Path("configs/phase3").glob("*.yaml")):
            c = load_config(p)
            if c["attack"].get("physical"):
                continue
            c["run_name"] += tag
            c["model_config"] = model_cfgs[c["model_config"]].as_posix()
            _dump(c, out / "phase3" / p.name)
        c = load_config("configs/phase4/transfer.yaml")
        c["run_name"] += tag
        c["transfer"]["models"] = [model_cfgs[m].as_posix() for m in c["transfer"]["models"]]
        c["transfer"].pop("physical_models", None)
        _dump(c, out / "phase4" / "transfer.yaml")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--defended", action="store_true")
    a = ap.parse_args()
    for s in a.seeds:
        clone(s, a.defended)
