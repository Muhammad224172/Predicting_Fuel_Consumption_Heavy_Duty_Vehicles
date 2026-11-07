import numpy as np
import pandas as pd
import pytest

from fuelcons.features import TARGET, add_targets, build_window_table
from fuelcons.selection import ORIGINAL_FEATURES, CorrelationSelector, feature_columns


def test_original_feature_set_matches_notebook():
    assert len(ORIGINAL_FEATURES) == 58 == len(set(ORIGINAL_FEATURES))
    assert not any(c.endswith("_median") for c in ORIGINAL_FEATURES)
    assert not any(c.startswith("slope_positive_") and not c.endswith("_start") for c in ORIGINAL_FEATURES)


@pytest.mark.parametrize("name", ["original", "no_median", "all"])
def test_feature_sets_exclude_target_components(name):
    cols = feature_columns(name)
    assert not any(c.startswith("EngTotalFuelUsed") for c in cols)
    assert [c for c in cols if c.startswith("HghRslution")] == ["HghRslutionTotalVehicleDistance_range"]
    assert TARGET not in cols and "fuel_per_weight" not in cols


def test_target_definition():
    df = pd.DataFrame(
        {
            "EngTotalFuelUsed_range": [0.5, 0.5],
            "HghRslutionTotalVehicleDistance_range": [10_000.0, 0.0],
            "GrossCombinationVehicleWeight_mean": [0.5, 0.5],
        }
    )
    out = add_targets(df)
    assert out[TARGET].iloc[0] == pytest.approx(5.0)  # 0.5 per 10 km = 5 per 100 km
    assert np.isinf(out[TARGET].iloc[1])
    assert out["fuel_per_weight"].iloc[0] == pytest.approx(1.0)


def test_build_window_table(raw_dir):
    table = build_window_table(raw_dir, "rows", 120, 40)
    assert table["record_id"].nunique() == 8
    assert table["vehicle_id"].nunique() == 4
    assert set(feature_columns("all")) <= set(table.columns)
    assert (table.groupby("record_id")["window_index"].diff().dropna() == 1).all()
    seconds = build_window_table(raw_dir, "seconds", 120, 40)
    assert len(seconds) > 0


def test_correlation_selector_keeps_informative_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = 3 * X[:, 4] - 2 * X[:, 1] + rng.normal(scale=0.1, size=300)
    sel = CorrelationSelector(k=2).fit(X, y)
    assert sel.support_.tolist() == [1, 4]
    assert sel.transform(X).shape == (300, 2)
