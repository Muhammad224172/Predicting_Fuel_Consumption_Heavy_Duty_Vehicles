"""Reproduce all results: ``python scripts/run_pipeline.py [--config ...] [--raw-dir ...]``.

Works without installing the package (adds ``src/`` to the import path).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuelcons.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
