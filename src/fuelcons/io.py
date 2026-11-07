"""Raw-data schema and loading of the per-record telemetry CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TIME_COL = "DateTime"
VEHICLE_COL = "VehicleID"

# The twelve CAN-bus signals in each record file (see data/README.md for units).
SIGNALS = [
    "HghRslutionTotalVehicleDistance",
    "TachographVehicleSpeed",
    "EngSpeed",
    "ActualEngPercentTorque",
    "AccelPedalPos1",
    "BrakePedalPos",
    "PCCM_Slope",
    "DStgy_dmRdcAgAct",
    "EngOilTemp1",
    "EngCoolantTemp",
    "GrossCombinationVehicleWeight",
    "EngTotalFuelUsed",
]
RAW_COLUMNS = [TIME_COL, VEHICLE_COL, *SIGNALS]

RECORD_PATTERN = re.compile(r"^Record(\d+)\.csv$")


def list_records(raw_dir: str | Path) -> list[Path]:
    """Return the ``Record<N>.csv`` files in ``raw_dir``, sorted by N.

    The original notebook iterated in ``os.listdir`` order, which is
    filesystem-dependent; sorting numerically makes row order (and thus the
    original protocol's hold-out split) reproducible.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Place Record<N>.csv files there (see data/README.md)."
        )
    files = [p for p in raw_dir.iterdir() if RECORD_PATTERN.match(p.name)]
    if not files:
        raise FileNotFoundError(f"No Record<N>.csv files in {raw_dir} (see data/README.md).")
    return sorted(files, key=record_id)


def record_id(path: str | Path) -> int:
    """Numeric id of a record file, e.g. ``Record17.csv`` -> 17."""
    match = RECORD_PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"Not a record file name: {path}")
    return int(match.group(1))


def read_record(path: str | Path) -> pd.DataFrame:
    """Read one record file, check its schema and parse timestamps."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[RAW_COLUMNS].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[SIGNALS] = df[SIGNALS].apply(pd.to_numeric, errors="coerce")
    if df[VEHICLE_COL].nunique() != 1:
        raise ValueError(f"{path}: expected exactly one VehicleID per record")
    return df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
