"""Rebuild any trained model (baseline or compression arm) from its run config. Used by Parts 3-5."""
import torch

from src.models.resnet import build_model


def load_model(cfg, device):
    model = build_model(cfg["model"], width=cfg.get("width", 64))
    c = cfg.get("compression") or {}
    if c.get("method") == "quant":
        from src.compress.quantize import quantize_model
        model = quantize_model(model, c["weight_bits"], c["act_bits"])
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    return model.to(device).eval()
