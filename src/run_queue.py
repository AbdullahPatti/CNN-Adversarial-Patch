"""Run a list of configs one after another, each in its own process, logging to logs/<phase>/<run_name>.log
(<phase> is the config's folder, e.g. configs/phase3/x.yaml -> logs/phase3/).

Runs whose results/<run_name>.json already exists are skipped, so the queue can be restarted
after an interruption and continues where it stopped.

Usage:  python -m src.run_queue configs/phase2/*.yaml
"""
import subprocess
import sys
import time
from pathlib import Path

from src.utils import load_config

ENTRY = {"none": "src.train", "prune": "src.compress.prune", "quant": "src.compress.quantize",
         "distill": "src.compress.distill"}


def main():
    for path in sys.argv[1:]:
        cfg = load_config(path)
        phase = Path(path).parent.name
        log_dir = Path("logs", phase) if phase.startswith("phase") else Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        run = cfg["run_name"]
        if Path("results", f"{run}.json").exists():
            print(f"[skip] {run}: results exist", flush=True)
            continue
        entry = "src.attack" if cfg.get("attack") else ENTRY[(cfg.get("compression") or {}).get("method", "none")]
        print(f"[start] {run}  ({entry})  {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        with open(log_dir / f"{run}.log", "w") as log:
            code = subprocess.call([sys.executable, "-u", "-m", entry, "--config", path],
                                   stdout=log, stderr=subprocess.STDOUT)
        status = "done" if code == 0 else f"FAILED (exit {code})"
        print(f"[{status}] {run}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("[queue finished]", flush=True)


if __name__ == "__main__":
    main()
