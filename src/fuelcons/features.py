"""Window aggregates and window-level targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean_record
from .io import SIGNALS, TIME_COL, VEHICLE_COL, list_records, read_record, record_id
from .windowing import make_windows

STATS = ("mean", "median", "max", "min", "std", "range")
# Signals aggregated per window: the raw signals plus the two slope indicators.
# (The original loop listed ``slope_negative`` twice and never aggregated
# ``slope_positive``; both are aggregated here, and the ``original`` feature
# set in selection.py leaves ``slope_positive_*`` out to stay faithful.)
AGG_SIGNALS = [*SIGNALS, "slope_positive", "slope_negative"]

ID_COLUMNS = ["record_id", "vehicle_id", "window_index", "start_time", "end_time", "n_rows"]
TARGET = "fuel_per_100km"


def window_features(record: pd.DataFrame, bounds: np.ndarray) -> pd.DataFrame:
    """One row per window: identifiers plus ``<signal>_<stat>`` aggregates.

    Also keeps the slope indicators at the first row of each window
    (``slope_*_start``), because the original model used those point values
    as two of its 58 features.
    """
    values = record[AGG_SIGNALS].to_numpy(dtype=float)
    times = record[TIME_COL].to_numpy()
    rows = []
    for i, (start, end) in enumerate(bounds):
        w = values[start:end]
        wmax, wmin = w.max(axis=0), w.min(axis=0)
        stats = {
            "mean": w.mean(axis=0),
            "median": np.median(w, axis=0),
            "max": wmax,
            "min": wmin,
            "std": w.std(axis=0),  # population std (ddof=0), as np.std in the original
            "range": wmax - wmin,
        }
        row = {
            "window_index": i,
            "start_time": times[start],
            "end_time": times[end - 1],
            "n_rows": end - start,
            "slope_positive_start": record["slope_positive"].iat[start],
            "slope_negative_start": record["slope_negative"].iat[start],
        }
        for stat in STATS:
            row.update({f"{sig}_{stat}": v for sig, v in zip(AGG_SIGNALS, stats[stat], strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def add_targets(table: pd.DataFrame) -> pd.DataFrame:
    """Add the window-level targets used in the original project.

    * ``fuel_per_100km``: fuel used in the window per 100 km driven
      (``EngTotalFuelUsed`` is masked by the data provider, so the unit is
      "masked fuel units / 100 km"; distance is in metres, hence the 1e5).
    * ``fuel_per_weight``: fuel used per unit of (masked) gross weight. It is
      only used for outlier removal, never as a feature.

    Windows with zero distance give inf/NaN; the IQR filter removes them.
    """
    out = table.copy()
    fuel = out["EngTotalFuelUsed_range"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[TARGET] = fuel * 1e5 / out["HghRslutionTotalVehicleDistance_range"]
        out["fuel_per_weight"] = fuel / out["GrossCombinationVehicleWeight_mean"]
    return out


def build_window_table(raw_dir: str | Path, unit: str, size: float, step: float) -> pd.DataFrame:
    """Clean every record in ``raw_dir``, window it and aggregate each window."""
    parts = []
    for path in list_records(raw_dir):
        record = clean_record(read_record(path))
        bounds = make_windows(record, unit, size, step, TIME_COL)
        if len(bounds) == 0:
            continue
        feats = window_features(record, bounds)
        feats.insert(0, "vehicle_id", int(record[VEHICLE_COL].iat[0]))
        feats.insert(0, "record_id", record_id(path))
        parts.append(feats)
    if not parts:
        raise ValueError("No record was long enough to produce a single window.")
    return add_targets(pd.concat(parts, ignore_index=True))
