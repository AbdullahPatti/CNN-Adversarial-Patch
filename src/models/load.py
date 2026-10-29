"""Rebuild any trained model (baseline or compression arm) from its run config. Used by Parts 3-5."""
import torch

from src.data import spec
from src.models.resnet import build_model


def build_from_cfg(cfg, width=None):
    """Untrained model of the config's architecture and dataset; width=None -> cfg["width"] or the standard."""
    ds = cfg.get("dataset", "cifar10")
    return build_model(cfg["model"], spec(ds)["num_classes"], width or cfg.get("width"), ds)


def load_model(cfg, device):
    model = build_from_cfg(cfg)
    c = cfg.get("compression") or {}
    if c.get("method") == "quant":
        from src.compress.quantize import quantize_model
        model = quantize_model(model, c["weight_bits"], c["act_bits"])
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    return model.to(device).eval()
