import numpy as np
import pandas as pd
import pytest

from fuelcons.features import AGG_SIGNALS, window_features
from fuelcons.windowing import row_windows, time_windows


@pytest.mark.parametrize("n,size,step", [(3000, 600, 120), (600, 600, 120), (601, 600, 120), (1000, 300, 60)])
def test_row_windows_match_original_loop(n, size, step):
    bounds = row_windows(n, size, step)
    original_starts = list(range(0, n - size + 1, step))  # notebook cell 15
    assert bounds[:, 0].tolist() == original_starts
    assert (bounds[:, 1] - bounds[:, 0] == size).all()
    assert len(bounds) == (n - size) // step + 1
    assert bounds[-1, 1] <= n


def test_row_windows_too_short_record():
    assert row_windows(599, 600, 120).shape == (0, 2)


def test_overlap_fraction():
    bounds = row_windows(2000, 600, 120)
    shared = bounds[0, 1] - bounds[1, 0]
    assert shared / 600 == pytest.approx(0.8)


def test_time_windows_equal_row_windows_at_1hz():
    times = pd.Series(pd.date_range("2015-01-01", periods=1000, freq="1s"))
    assert np.array_equal(time_windows(times, 300, 60), row_windows(1000, 300, 60))


def test_time_windows_respect_wall_clock_on_irregular_sampling():
    rng = np.random.default_rng(0)
    times = pd.Series(pd.Timestamp("2015-01-01") + pd.to_timedelta(np.cumsum(rng.choice([0.2, 1, 2], 2000)), "s"))
    t = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    bounds = time_windows(times, 300, 60)
    assert len(bounds) > 0 and bounds[0, 0] == 0
    assert (np.diff(bounds[:, 0]) > 0).all()
    for start, end in bounds:
        assert t[end - 1] - t[start] < 300  # every sample lies inside one 300 s span
        assert end == len(t) or t[end] - t[start] >= 300 - 2  # and the window is not cut short


def _original_extract_features(window):  # notebook cell 14
    return {
        "mean": np.mean(window),
        "median": np.median(window),
        "max": np.max(window),
        "min": np.min(window),
        "std": np.std(window),
        "range": np.max(window) - np.min(window),
    }


def test_window_features_match_original_definition():
    rng = np.random.default_rng(1)
    n = 700
    record = pd.DataFrame({sig: rng.normal(size=n) for sig in AGG_SIGNALS})
    record["DateTime"] = pd.date_range("2015-01-01", periods=n, freq="1s")
    bounds = row_windows(n, 600, 50)
    feats = window_features(record, bounds)
    assert len(feats) == len(bounds)
    for i, (start, end) in enumerate(bounds):
        for sig in ("EngSpeed", "slope_negative"):
            expected = _original_extract_features(record[sig].to_numpy()[start:end])
            for stat, value in expected.items():
                assert feats.loc[i, f"{sig}_{stat}"] == pytest.approx(value)
        assert feats.loc[i, "slope_positive_start"] == record["slope_positive"].iat[start]
