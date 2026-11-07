import numpy as np
import pandas as pd

from fuelcons.cleaning import clean_record, iqr_mask, remove_outliers
from fuelcons.io import SIGNALS, list_records, read_record, record_id


def test_list_records_sorted_numerically(raw_dir):
    ids = [record_id(p) for p in list_records(raw_dir)]
    assert ids == sorted(ids) and len(ids) == 8


def test_clean_record_fills_all_signals(raw_dir):
    raw = read_record(raw_dir / "Record3.csv")
    clean = clean_record(raw)
    assert not clean[SIGNALS].isna().any().any()
    # rows before every signal has been observed once are dropped, nothing else
    first_complete = max(raw[c].first_valid_index() for c in SIGNALS if c != "BrakePedalPos")
    assert len(clean) == len(raw) - first_complete
    assert set(clean["slope_positive"]) <= {0, 1} and set(clean["slope_negative"]) <= {0, 1}
    assert not ((clean["slope_positive"] == 1) & (clean["slope_negative"] == 1)).any()


def test_brake_pedal_missing_means_zero():
    df = pd.DataFrame({c: np.arange(4.0) for c in SIGNALS})
    df["BrakePedalPos"] = [10.0, np.nan, np.nan, 20.0]
    out = clean_record(df)
    assert out["BrakePedalPos"].tolist() == [10.0, 0.0, 0.0, 20.0]  # filled, not interpolated


def test_counters_are_interpolated_linearly():
    df = pd.DataFrame({c: np.arange(5.0) for c in SIGNALS})
    df["EngTotalFuelUsed"] = [100.0, np.nan, np.nan, np.nan, 104.0]
    out = clean_record(df)
    assert np.allclose(out["EngTotalFuelUsed"], [100, 101, 102, 103, 104])


def test_iqr_mask_drops_inf_and_nan():
    s = pd.Series([*range(1, 11), np.inf, np.nan, 100.0], dtype=float)
    assert iqr_mask(s).tolist() == [True] * 10 + [False, False, False]


def test_remove_outliers_is_sequential_and_logged():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100], "b": [1, 2, 3, 50, 4]})
    out, log = remove_outliers(df, ["a", "b"])
    assert [entry["removed"] for entry in log] == [1, 1]
    assert out["a"].tolist() == [1, 2, 3]
