"""Per-record cleaning and window-level outlier removal.

The per-record steps reproduce notebook cells 4-12 of the original project:
brake pedal NaN -> 0, linear interpolation, forward fill, drop the remaining
(leading) incomplete rows, then add uphill/downhill indicators.
"""

from __future__ import annotations

import pandas as pd

from .io import SIGNALS


def clean_record(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in one record so that every row has every signal.

    Signals are logged at different rates (the distance, weight and fuel
    counters appear in only ~2-4% of rows), so most cells are empty.

    1. ``BrakePedalPos`` is only logged while the pedal is pressed, so a
       missing value means 0.
    2. Remaining gaps are linearly interpolated over row position (as in the
       original; rows are ~1 s apart) and trailing gaps are forward-filled.
    3. Leading rows recorded before a signal's first observation cannot be
       filled and are dropped.
    """
    out = df.copy()
    out["BrakePedalPos"] = out["BrakePedalPos"].fillna(0.0)
    out[SIGNALS] = out[SIGNALS].interpolate(method="linear").ffill()
    out = out.dropna(subset=SIGNALS).reset_index(drop=True)
    out["slope_positive"] = (out["PCCM_Slope"] > 0).astype(int)
    out["slope_negative"] = (out["PCCM_Slope"] < 0).astype(int)
    return out


def iqr_mask(values: pd.Series, k: float = 1.5) -> pd.Series:
    """True for values inside ``[Q1 - k*IQR, Q3 + k*IQR]``.

    NaN and +/-inf fall outside the interval and are therefore removed, which
    is how the original notebook implicitly dropped windows with zero distance.
    """
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    return (values >= q1 - k * iqr) & (values <= q3 + k * iqr)


def remove_outliers(table: pd.DataFrame, columns: list[str], k: float = 1.5) -> tuple[pd.DataFrame, list[dict]]:
    """Apply IQR filters to ``columns`` one after another (order matters).

    Returns the filtered table and a log of how many windows each step removed.
    """
    log = []
    for col in columns:
        before = len(table)
        table = table[iqr_mask(table[col], k)]
        log.append({"column": col, "removed": before - len(table), "remaining": len(table)})
    return table.reset_index(drop=True), log
