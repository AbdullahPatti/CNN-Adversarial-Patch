"""Run a list of configs one after another, each in its own process, logging to logs/<run_name>.log.

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
    Path("logs").mkdir(exist_ok=True)
    for path in sys.argv[1:]:
        cfg = load_config(path)
        run = cfg["run_name"]
        if Path("results", f"{run}.json").exists():
            print(f"[skip] {run}: results exist", flush=True)
            continue
        entry = ENTRY[(cfg.get("compression") or {}).get("method", "none")]
        print(f"[start] {run}  ({entry})  {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        with open(Path("logs", f"{run}.log"), "w") as log:
            code = subprocess.call([sys.executable, "-u", "-m", entry, "--config", path],
                                   stdout=log, stderr=subprocess.STDOUT)
        status = "done" if code == 0 else f"FAILED (exit {code})"
        print(f"[{status}] {run}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("[queue finished]", flush=True)


if __name__ == "__main__":
    main()
