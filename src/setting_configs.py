"""Configs for the follow-up phases (J5-J7, J9), approved 2026-10-14 / 2026-10-20.

Second settings, each with the baseline and the four 4x arms (INT8 PTQ, 75% pruning, dense-small, KD):
  gtsrb   ResNet-18 on GTSRB (32 px, 43 classes)                        J5
  mbv2    MobileNetV2 on CIFAR-10, 120 epochs (200 would cost ~46 GPU-h for 3 seeds)   J7
  inette  ResNet-50 on ImageNette at 224 px (patch sides 35 / 56 = the same 2.44% / 6.25% areas as 5 / 8 at 32 px),
          one seed; LaVAN and certification on the first 500 images of the fixed subset (13x the per-image cost)  J6
Per seed: configs/<setting>/seed<S>/{phase2,phase3,phase4,phase5} (models, universal + LaVAN attacks, transfer,
certification) and configs/<setting>/seed<S>_def/{phase2,phase5} (defended zoo: mask augmentation in every training
run, defend-then-compress as in Part 5, then certified). Run names <setting>_<arm>_s<S>[_def]. Every run keeps the
final epoch (train.select = last), so the test set plays no part in training. Attack seed 0 everywhere, as in Part 3.

J9 (attack completeness, CIFAR-10 ResNet-18, seeds 42-44): configs/j9/:
  universal_untargeted_<model>  untargeted universal patch (maximize the true-label loss), all 13 models
  lavan_margin_<model>          LaVAN with the Carlini-Wagner margin loss instead of CE, baseline + 4x arms

Usage:  python -m src.setting_configs
"""
from pathlib import Path

import torch
import yaml

from src.seed_configs import MASK_AUG

SEEDS = (42, 43, 44)

SETTINGS = {
    "gtsrb": {"model": "resnet18", "dataset": "gtsrb", "small_width": 32, "epochs": 100, "ft_epochs": 20,
              "batch_size": 128, "lr": 0.1, "num_workers": 8, "sizes": [5, 8], "img": 32,
              # 10 fixed target classes for the universal patch (43 targets would cost 4.3x)
              "targets": sorted(torch.randperm(43, generator=torch.Generator().manual_seed(0))[:10].tolist()),
              "attack_batch": 128, "lavan_batch": 500, "pc_batch": 1000, "seeds": SEEDS},
    "mbv2": {"model": "mobilenetv2", "dataset": "cifar10", "small_width": 16, "epochs": 120, "ft_epochs": 30,
             "batch_size": 128, "lr": 0.1, "num_workers": 8, "sizes": [5, 8], "img": 32, "targets": list(range(10)),
             "attack_batch": 128, "lavan_batch": 500, "pc_batch": 1000, "seeds": SEEDS},
    "inette": {"model": "resnet50", "dataset": "imagenette", "small_width": 32, "epochs": 120, "ft_epochs": 30,
               "batch_size": 64, "lr": 0.05, "num_workers": 12, "sizes": [35, 56], "img": 224,
               "targets": list(range(10)), "attack_batch": 32, "lavan_batch": 50, "pc_batch": 100, "seeds": (42,), "n_images": 500},
}
ARMS = ["baseline", "quant_4x_int8", "prune_4x", "dense_small_4x", "kd_4x"]


def _dump(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def model_cfg(name, s, arm, seed, defended):
    tag = f"_s{seed}" + ("_def" if defended else "")
    run = f"{name}_{arm}{tag}"
    base_ckpt = f"checkpoints/{name}/{name}_baseline{tag}.pt"
    train = {"epochs": s["epochs"], "batch_size": s["batch_size"], "lr": s["lr"], "momentum": 0.9,
             "weight_decay": 5.0e-4, "select": "last"}
    c = {"run_name": run, "model": s["model"], "dataset": s["dataset"], "seed": seed, "data_root": "./data",
         "num_workers": s["num_workers"]}
    if arm == "baseline":
        c.update(checkpoint=f"checkpoints/{name}/{run}.pt", compression={"method": "none", "ratio": 1.0})
    elif arm == "quant_4x_int8":
        c.update(base_checkpoint=base_ckpt, checkpoint=f"checkpoints/{name}/{run}.pt",
                 compression={"method": "quant", "ratio": 4.0, "mode": "ptq", "weight_bits": 8, "act_bits": 8,
                              "calib_batches": 20})
    elif arm == "prune_4x":
        c.update(base_checkpoint=base_ckpt, checkpoint=f"checkpoints/{name}/{run}.pt",
                 compression={"method": "prune", "ratio": 4.0, "sparsity": 0.75})
        train.update(epochs=s["ft_epochs"], lr=s["lr"] / 10)
    elif arm == "dense_small_4x":
        c.update(width=s["small_width"], checkpoint=f"checkpoints/{name}/{run}.pt",
                 compression={"method": "none", "ratio": 4.0, "note": "capacity control for RQ4, no compression applied"})
    elif arm == "kd_4x":
        c.update(width=s["small_width"], base_checkpoint=base_ckpt, checkpoint=f"checkpoints/{name}/{run}.pt",
                 compression={"method": "distill", "ratio": 4.0, "temperature": 4.0, "alpha": 0.9})
    if defended:
        train["mask_aug"] = {**MASK_AUG, "sizes": s["sizes"], "img": s["img"]}
    c["train"] = train
    return c


def setting(name, seed):
    s = SETTINGS[name]
    for defended in (False, True):
        out = Path("configs", name, f"seed{seed}" + ("_def" if defended else ""))
        paths = []
        for i, arm in enumerate(ARMS):
            c = model_cfg(name, s, arm, seed, defended)
            p = out / "phase2" / f"{i:02d}_{arm}.yaml"
            _dump(c, p)
            paths.append(p.as_posix())
            _dump({"run_name": f"pc_{c['run_name']}", "model_config": p.as_posix(), "seed": 0, "data_root": "./data",
                   "num_workers": 0, "defense": {"name": "patchcleanser", "sizes": s["sizes"], "k": MASK_AUG["k"],
                                                 "fine_tuned": defended, "batch": s["pc_batch"],
                                                 **({"n_images": s["n_images"]} if "n_images" in s else {})}},
                  out / "phase5" / f"{i:02d}_pc_{arm}.yaml")
            if defended:
                continue
            quant = arm.startswith("quant")
            _dump({"run_name": f"universal_{c['run_name']}", "model_config": p.as_posix(), "seed": 0,
                   "data_root": "./data", "num_workers": s["num_workers"],
                   "attack": {"name": "universal", "sizes": s["sizes"], "targets": s["targets"], "steps": 1000,
                              "lr": 0.02, "batch_size": s["attack_batch"], "monitor_every": 250, "physical": False,
                              "surrogate": quant}},
                  out / "phase3" / f"{i:02d}_universal_{arm}.yaml")
            _dump({"run_name": f"lavan_{c['run_name']}", "model_config": p.as_posix(), "seed": 0,
                   "data_root": "./data", "num_workers": s["num_workers"],
                   "attack": {"name": "lavan", "sizes": s["sizes"], "modes": ["untargeted", "targeted"], "steps": 100,
                              "step_size": 0.05, "restarts": 10, "batch": s["lavan_batch"], "surrogate": quant,
                              **({"n_images": s["n_images"]} if "n_images" in s else {})}},
                  out / "phase3" / f"{10 + i:02d}_lavan_{arm}.yaml")
        if not defended:
            _dump({"run_name": f"transfer_{name}_s{seed}", "seed": 0, "data_root": "./data", "num_workers": 0,
                   "transfer": {"models": paths, **({"n_images": s["n_images"]} if "n_images" in s else {})}},
                  out / "phase4" / "transfer.yaml")
    print(f"wrote configs/{name}/seed{seed}[_def]")


def j9():
    """Attack-completeness configs on the existing CIFAR-10 ResNet-18 models of all three seeds."""
    out = Path("configs", "j9")
    for seed in SEEDS:
        tag = "" if seed == 42 else f"_s{seed}"
        src = Path("configs/phase3") if seed == 42 else Path(f"configs/seed{seed}/phase3")
        for p in sorted(src.glob("*_universal_*.yaml")):
            if "physical" in p.name:
                continue
            c = yaml.safe_load(p.read_text())
            c["run_name"] = c["run_name"].replace("universal_", "universal_untargeted_", 1)
            c["attack"]["targets"] = [-1]
            _dump(c, out / f"seed{seed}" / p.name.replace("universal", "universal_untargeted"))
        for p in sorted(src.glob("*_lavan_*.yaml")):
            if not any(a in p.name for a in ("baseline", "_4x")):
                continue
            c = yaml.safe_load(p.read_text())
            c["run_name"] = c["run_name"].replace("lavan_", "lavan_margin_", 1)
            c["attack"]["loss"] = "margin"
            _dump(c, out / f"seed{seed}" / p.name.replace("lavan", "lavan_margin"))
        assert all(tag in yaml.safe_load(p.read_text())["run_name"] for p in (out / f"seed{seed}").glob("*.yaml"))
    print("wrote configs/j9")


if __name__ == "__main__":
    for n in SETTINGS:
        for sd in SETTINGS[n]["seeds"]:
            setting(n, sd)
    j9()
