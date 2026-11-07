"""Synthetic telemetry that mimics the raw schema (for tests only).

The numbers are invented: they exercise the code paths (irregular sampling,
sparse signals, cumulative counters, leading gaps) but say nothing about real
trucks. Never report metrics computed on this data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fuelcons.io import RAW_COLUMNS

# Approximate share of rows in which each signal is logged in the real data.
OBSERVED_FRACTION = {
    "HghRslutionTotalVehicleDistance": 0.04,
    "TachographVehicleSpeed": 0.2,
    "EngSpeed": 0.5,
    "ActualEngPercentTorque": 0.2,
    "AccelPedalPos1": 0.2,
    "PCCM_Slope": 0.35,
    "DStgy_dmRdcAgAct": 0.2,
    "EngOilTemp1": 0.04,
    "EngCoolantTemp": 0.2,
    "GrossCombinationVehicleWeight": 0.03,
    "EngTotalFuelUsed": 0.03,
}


def make_record(vehicle_id: int, n_rows: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 2.0], size=n_rows)
    start = pd.Timestamp("2015-10-01 08:00:00") + pd.Timedelta(days=int(seed))
    t = start + pd.to_timedelta(np.cumsum(dt), unit="s")

    slope = np.clip(np.cumsum(rng.normal(0, 0.05, n_rows)), -3, 3)
    speed = np.clip(75 + np.cumsum(rng.normal(0, 0.3, n_rows)), 0, 90) * (rng.random() > 0.05)
    torque = np.clip(35 + 10 * slope + rng.normal(0, 5, n_rows), 0, 100)
    weight = 0.3 + 0.6 * ((vehicle_id * 37) % 10) / 10
    fuel_rate = 1e-3 * (0.2 + torque / 100 * (1 + weight)) * (speed > 0)
    df = pd.DataFrame(
        {
            "DateTime": t.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
            "VehicleID": vehicle_id,
            "HghRslutionTotalVehicleDistance": 2e8 + np.cumsum(speed / 3.6 * dt),
            "TachographVehicleSpeed": speed,
            "EngSpeed": 600 + 7 * speed + rng.normal(0, 20, n_rows),
            "ActualEngPercentTorque": torque,
            "AccelPedalPos1": np.clip(0.9 * torque + rng.normal(0, 3, n_rows), 0, 100),
            "BrakePedalPos": np.where(rng.random(n_rows) < 0.05, rng.uniform(5, 40, n_rows), np.nan),
            "PCCM_Slope": np.round(slope * 4) / 4,
            "DStgy_dmRdcAgAct": np.clip(500 * fuel_rate + rng.normal(0, 0.05, n_rows), 0, None),
            "EngOilTemp1": np.linspace(0.4, 0.6, n_rows),
            "EngCoolantTemp": np.linspace(0.5, 0.7, n_rows),
            "GrossCombinationVehicleWeight": weight + rng.normal(0, 1e-3, n_rows),
            "EngTotalFuelUsed": 1e4 + np.cumsum(fuel_rate * dt),
        }
    )
    for col, frac in OBSERVED_FRACTION.items():
        df.loc[rng.random(n_rows) > frac, col] = np.nan
        df.loc[: rng.integers(1, 20), col] = np.nan  # leading gap before first observation
    return df[RAW_COLUMNS]


def write_raw_dir(path: Path, n_records: int = 8, n_vehicles: int = 4, n_rows: int = 900) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_records + 1):
        make_record(vehicle_id=1 + (i - 1) % n_vehicles, n_rows=n_rows, seed=i).to_csv(
            path / f"Record{i}.csv", index=False, na_rep="NaN"
        )
    return path
