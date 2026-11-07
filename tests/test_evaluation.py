import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from fuelcons.evaluation import cross_validate, make_splitter, regression_metrics


@pytest.fixture
def grouped_data():
    rng = np.random.default_rng(0)
    n = 400
    records = np.repeat(np.arange(40), 10)  # 40 records x 10 windows
    vehicles = records // 4  # 10 vehicles
    X = pd.DataFrame({"row_id": np.arange(n), "x": rng.normal(size=n)})
    y = pd.Series(2 * X["x"] + rng.normal(scale=0.1, size=n))
    return X, y, records, vehicles


@pytest.mark.parametrize("which", ["records", "vehicles"])
def test_grouped_cv_never_shares_a_group(grouped_data, which):
    X, y, records, vehicles = grouped_data
    groups = records if which == "records" else vehicles
    tested = []
    for train, test in make_splitter("group_record", 5, seed=0).split(X, y, groups):
        assert not set(groups[train]) & set(groups[test])
        tested.extend(test)
    assert sorted(tested) == list(range(len(X)))  # every window tested exactly once


def test_shuffled_kfold_mixes_records(grouped_data):
    """Documents the leakage the grouped schemes remove."""
    X, y, records, _ = grouped_data
    train, test = next(make_splitter("kfold", 5, seed=0).split(X))
    assert set(records[train]) & set(records[test])


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Mean predictor that records which rows it was fitted on."""

    fitted_rows: list = []

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        SpyRegressor.fitted_rows.append(set(X["row_id"]))
        self.mean_ = float(np.mean(y)) + self.alpha
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@pytest.mark.parametrize("scheme", ["kfold", "group_record"])
def test_nested_tuning_only_sees_training_rows(grouped_data, scheme):
    X, y, records, _ = grouped_data
    groups = None if scheme == "kfold" else records
    outer = list(make_splitter(scheme, 4, seed=1).split(X, y, groups))
    SpyRegressor.fitted_rows = []
    folds, preds = cross_validate(
        X,
        y,
        groups,
        SpyRegressor(),
        {"alpha": [0.0, 1.0]},
        scheme,
        n_splits=4,
        inner_splits=3,
        tuning="nested",
        seed=1,
        n_jobs=1,
    )
    fits_per_fold = 2 * 3 + 1  # grid x inner folds + refit on the outer training fold
    assert len(SpyRegressor.fitted_rows) == 4 * fits_per_fold
    for k, (train, test) in enumerate(outer):
        for rows in SpyRegressor.fitted_rows[k * fits_per_fold : (k + 1) * fits_per_fold]:
            assert rows <= set(train) and not rows & set(test)
    assert sorted(preds["row"]) == list(range(len(X)))
    assert all(json.loads(f["params"]) in ({"alpha": 0.0}, {"alpha": 1.0}) for f in folds)


def test_original_tuning_uses_first_80_percent(grouped_data):
    X, y, _, _ = grouped_data
    SpyRegressor.fitted_rows = []
    cross_validate(
        X,
        y,
        None,
        SpyRegressor(),
        {"alpha": [0.0, 1.0]},
        "kfold",
        n_splits=4,
        inner_splits=3,
        tuning="original",
        seed=0,
        n_jobs=1,
    )
    tuning_fits = SpyRegressor.fitted_rows[: 2 * 5 + 1]
    assert set().union(*tuning_fits) == set(range(int(0.8 * len(X))))


def test_cross_validate_is_deterministic(grouped_data):
    from fuelcons.models import make_model

    X, y, records, _ = grouped_data
    est, _ = make_model("xgboost", seed=0)
    grid = {"model__n_estimators": [20], "model__max_depth": [2, 3]}
    runs = [
        cross_validate(
            X[["x"]],
            y,
            records,
            est,
            grid,
            "group_record",
            n_splits=3,
            inner_splits=2,
            tuning="nested",
            seed=0,
            n_jobs=1,
        )[0]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_regression_metrics():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert m["r2"] == pytest.approx(1 - 4 / 2)
