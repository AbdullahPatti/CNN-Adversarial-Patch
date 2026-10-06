"""Config, seeding, and the shared results-JSON writer."""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_result(run_name, config, metrics, out_dir="results"):
    """One JSON per run, same schema for every experiment (plan, Sec. 3).

    Paper tables and the frontend dashboard both read these files.
    Later parts add attack / defense fields under the same top-level keys.
    """
    record = {
        "run": run_name,
        "model": config["model"],
        "compression": config.get("compression", {"method": "none", "ratio": 1.0}),
        "attack": config.get("attack"),
        "defense": config.get("defense"),
        "metrics": metrics,
        "config": config,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{run_name}.json"
    path.write_text(json.dumps(record, indent=2))
    return path
