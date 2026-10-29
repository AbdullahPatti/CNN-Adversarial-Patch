"""Part 5: PatchCleanser certified accuracy (RQ3) of one trained model on the fixed 1,000-image subset.

Per patch size reports: certified accuracy, clean accuracy of the defended model (double masking), clean
accuracy without the defense, and clean accuracy under the masks alone (mean / worst over the one-mask set,
mean over all two-mask pairs). The mask-only numbers separate "this model can't handle occlusion" from
"the defense fails". Per-image certified flags are kept for paired statistics across models and seeds.

Usage:  python -m src.certify --config configs/phase5/pc_kd_4x.yaml
"""
import argparse

import torch

from src.attack import accuracy, stack_dataset
from src.attacks.patch import eval_batch
from src.data import eval_subset_indices, get_loaders
from src.defense.patchcleanser import certified, double_masking, mask_positions, mask_set, two_mask_preds
from src.models.load import load_model
from src.utils import get_device, load_config, save_result


def certify_model(model, x, y, sizes, k, bs=1000):
    out = {}
    for p in sizes:
        pos, m = mask_positions(x.size(-1), p, k)
        masks = mask_set(x.size(-1), p, k, x.device)
        p2 = two_mask_preds(model, x, masks, model.norm.mean.flatten().tolist(), bs)
        ok = p2 == y.view(-1, 1, 1)
        one = ok.diagonal(dim1=1, dim2=2).float().mean(0)  # accuracy under each single mask
        cert = certified(p2, y)
        out[f"side_{p}"] = {
            "area_frac": p * p / x.size(-1) ** 2, "masks": len(masks), "mask_side": m, "mask_stride_positions": pos,
            "certified_acc": cert.float().mean().item(),
            "defended_clean_acc": (double_masking(p2) == y).float().mean().item(),
            "one_mask_acc_mean": one.mean().item(), "one_mask_acc_worst": one.min().item(),
            "two_mask_acc_mean": ok.float().mean().item(),
            "certified_per_image": cert.int().tolist(),
        }
        r = out[f"side_{p}"]
        print(f"side {p}: {len(masks)} masks of {m}px  certified {r['certified_acc']:.4f}  "
              f"defended clean {r['defended_clean_acc']:.4f}  one-mask {r['one_mask_acc_mean']:.4f} "
              f"(worst {r['one_mask_acc_worst']:.4f})  two-mask {r['two_mask_acc_mean']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = load_config(cfg["model_config"])
    cfg["model"], cfg["compression"] = mcfg["model"], mcfg.get("compression")
    ds = cfg["dataset"] = mcfg.get("dataset", "cifar10")
    d, device = cfg["defense"], get_device()
    torch.backends.cudnn.benchmark = True

    model = load_model(mcfg, device).requires_grad_(False)
    _, test_loader = get_loaders(cfg["data_root"], 128, 0, ds)
    x, y = stack_dataset(test_loader.dataset, device, eval_subset_indices(ds)[:d.get("n_images")])
    bs = d.get("batch", eval_batch(x))
    metrics = {"model_run": mcfg["run_name"], "clean_acc_subset": accuracy(model, x, y, bs),
               **certify_model(model, x, y, d["sizes"], d["k"], bs)}
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
