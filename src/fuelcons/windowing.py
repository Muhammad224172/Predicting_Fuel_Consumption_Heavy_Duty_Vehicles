"""Sliding-window segmentation of a cleaned record.

Two units are supported:

* ``rows``: windows of ``size`` consecutive rows, advanced by ``step`` rows.
  This is what the original notebook did (``window_size = 10 * 60`` rows,
  ``shift = 2 * 60`` rows). The raw logs are irregularly sampled (0.2-2 s,
  median 1 s), so 600 rows correspond to roughly, not exactly, 10 minutes.
* ``seconds``: windows of ``size`` seconds of wall-clock time, advanced by
  ``step`` seconds, located with the timestamps.

Both return half-open row-index intervals ``[start, end)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def row_windows(n_rows: int, size: int, step: int) -> np.ndarray:
    """Row-based windows; same start indices as ``range(0, n - size + 1, step)``."""
    _check(size, step)
    if n_rows < size:
        return np.empty((0, 2), dtype=int)
    starts = np.arange(0, n_rows - size + 1, step)
    return np.column_stack([starts, starts + size])


def time_windows(times: pd.Series, size: float, step: float, min_rows: int = 2) -> np.ndarray:
    """Time-based windows of ``size`` seconds that lie fully inside the record.

    Windows with fewer than ``min_rows`` samples (logging gaps) are skipped.
    """
    _check(size, step)
    t = (pd.to_datetime(times) - pd.to_datetime(times).iloc[0]).dt.total_seconds().to_numpy()
    if len(t) == 0 or t[-1] < size:
        return np.empty((0, 2), dtype=int)
    win_starts = np.arange(0.0, t[-1] - size + 1e-9, step)
    starts = np.searchsorted(t, win_starts, side="left")
    ends = np.searchsorted(t, win_starts + size, side="left")
    keep = (ends - starts) >= min_rows
    return np.column_stack([starts[keep], ends[keep]])


def make_windows(record: pd.DataFrame, unit: str, size: float, step: float, time_col: str) -> np.ndarray:
    if unit == "rows":
        return row_windows(len(record), int(size), int(step))
    if unit == "seconds":
        return time_windows(record[time_col], float(size), float(step))
    raise ValueError(f"Unknown window unit {unit!r}; use 'rows' or 'seconds'.")


def _check(size: float, step: float) -> None:
    if size <= 0 or step <= 0:
        raise ValueError("Window size and step must be positive.")
