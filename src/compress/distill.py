"""Knowledge distillation (Hinton et al. 2015) from the baseline into a narrower ResNet-18.

loss = alpha * T^2 * KL(student_T || teacher_T) + (1 - alpha) * CE(student, y)
The student trains from scratch with the baseline recipe; its dense-small twin (same width, plain
CE, src.train) separates what KD adds from what the smaller capacity costs.

Usage:  python -m src.compress.distill --config configs/phase2/kd_4x.yaml
"""
import argparse

import torch
import torch.nn.functional as F

from src.data import get_loaders
from src.engine import final_metrics, fit
from src.eval.metrics import clean_accuracy, compressed_size
from src.models.resnet import build_model
from src.utils import get_device, load_config, save_result, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    t, c = cfg["train"], cfg["compression"]
    T, alpha = c["temperature"], c["alpha"]

    train_loader, test_loader = get_loaders(cfg["data_root"], t["batch_size"], cfg["num_workers"])
    teacher = build_model(cfg["model"]).to(device)
    teacher.load_state_dict(torch.load(cfg["base_checkpoint"], map_location=device))
    teacher.eval()
    print(f"teacher acc {clean_accuracy(teacher, test_loader, device):.4f}", flush=True)
    student = build_model(cfg["model"], width=cfg["width"]).to(device)

    def kd_loss(model, x, y):
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = model(x)
        kl = F.kl_div(F.log_softmax(s_logits.float() / T, 1), F.softmax(t_logits.float() / T, 1),
                      reduction="batchmean")
        return alpha * T * T * kl + (1 - alpha) * F.cross_entropy(s_logits, y)

    fit(student, train_loader, test_loader, device, t, cfg["checkpoint"], loss_fn=kd_loss)

    metrics = {**final_metrics(student, test_loader, device), **compressed_size(student)}
    print(metrics)
    print("saved", save_result(cfg["run_name"], cfg, metrics))


if __name__ == "__main__":
    main()
