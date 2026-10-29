"""Train a dense model from scratch: the full-precision baseline and the dense-small controls (RQ4).

Usage:  python -m src.train --config configs/baseline.yaml
"""
import argparse

from src.data import get_loaders
from src.engine import final_metrics, fit
from src.eval.metrics import compressed_size
from src.models.load import build_from_cfg
from src.utils import get_device, load_config, save_result, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    t = cfg["train"]

    train_loader, test_loader = get_loaders(cfg["data_root"], t["batch_size"], cfg["num_workers"],
                                            cfg.get("dataset", "cifar10"))
    model = build_from_cfg(cfg).to(device)
    fit(model, train_loader, test_loader, device, t, cfg["checkpoint"])

    metrics = {**final_metrics(model, test_loader, device, cfg), **compressed_size(model, cfg=cfg)}
    print(metrics)
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
