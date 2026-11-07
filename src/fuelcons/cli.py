"""Command-line entry point.

    python scripts/run_pipeline.py                     # original_protocol.yaml, then default.yaml
    python scripts/run_pipeline.py --config configs/default.yaml --raw-dir /path/to/Records
    FUELCONS_RAW_DIR=/path/to/Records python scripts/run_pipeline.py

The raw-data folder is taken from --raw-dir, else $FUELCONS_RAW_DIR, else the
config's ``raw_dir`` (data/raw/).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import load_config
from .pipeline import run

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR_ENV = "FUELCONS_RAW_DIR"
DEFAULT_CONFIGS = [REPO_ROOT / "configs" / "original_protocol.yaml", REPO_ROOT / "configs" / "default.yaml"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fuel-consumption pipeline (ENS 492 refactor).")
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="YAML config; repeat to run several. Default: original_protocol.yaml then default.yaml.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help=f"Folder of Record<N>.csv files. Overrides ${RAW_DIR_ENV} and the config's raw_dir (default data/raw).",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the cached window table.")
    parser.add_argument("--n-jobs", type=int, help="Override the config's n_jobs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for path in args.config or DEFAULT_CONFIGS:
        cfg = load_config(path)
        # Relative paths in configs are relative to the repository root.
        for attr in ("raw_dir", "processed_dir", "results_dir"):
            setattr(cfg, attr, str(REPO_ROOT / getattr(cfg, attr)))
        if cfg.record_order_file:
            cfg.record_order_file = str(REPO_ROOT / cfg.record_order_file)
        if args.n_jobs is not None:
            cfg.n_jobs = args.n_jobs
        raw_dir = args.raw_dir or os.environ.get(RAW_DIR_ENV) or None
        run(cfg, raw_dir=raw_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
