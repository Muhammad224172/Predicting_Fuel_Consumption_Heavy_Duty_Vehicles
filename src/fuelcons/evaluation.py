"""Cross-validation schemes, hyper-parameter tuning and metrics.

Why several schemes: consecutive windows of the same record overlap heavily
(600-row windows advanced by 120 rows share 80% of their samples). With a
shuffled ``KFold`` (the original protocol) almost every test window has
near-duplicate neighbours in the training folds, so the score measures
interpolation within known trips. ``group_record`` keeps each record (trip)
in a single fold; ``group_vehicle`` keeps each truck in a single fold, which
is the closest to "a new truck on a new trip".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, train_test_split

GROUP_COLUMNS = {"kfold": None, "group_record": "record_id", "group_vehicle": "vehicle_id"}


def make_splitter(scheme: str, n_splits: int, seed: int):
    if scheme == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if scheme in ("group_record", "group_vehicle"):
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unknown CV scheme {scheme!r}; choose from {sorted(GROUP_COLUMNS)}")


def scheme_groups(table: pd.DataFrame, scheme: str) -> np.ndarray | None:
    col = GROUP_COLUMNS[scheme]
    return None if col is None else table[col].to_numpy()


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tune_like_original(estimator, grid: dict, X: pd.DataFrame, y: pd.Series, n_jobs: int) -> dict:
    """Original protocol: grid search on the first 80% of rows (no shuffle),
    5-fold unshuffled inner CV, default R^2 scoring. The chosen parameters are
    then re-used in every outer fold, so the tuning data overlaps the outer
    test folds; this is kept only to reproduce the original numbers."""
    X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    search = GridSearchCV(clone(estimator), grid, cv=5, n_jobs=n_jobs)
    search.fit(X_tune, y_tune)
    return search.best_params_


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None,
    estimator,
    grid: dict,
    scheme: str,
    *,
    n_splits: int,
    inner_splits: int,
    tuning: str,
    seed: int,
    n_jobs: int,
) -> tuple[list[dict], pd.DataFrame]:
    """Outer CV with tuning either nested in each training fold or done once
    up front (``tuning="original"``). Returns per-fold metrics and
    out-of-fold predictions."""
    if tuning not in ("nested", "original"):
        raise ValueError(f"Unknown tuning mode {tuning!r}")
    outer = make_splitter(scheme, n_splits, seed)
    fixed_params = tune_like_original(estimator, grid, X, y, n_jobs) if grid and tuning == "original" else {}

    folds, preds = [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if grid and tuning == "nested":
            inner = make_splitter(scheme, inner_splits, seed)
            search = GridSearchCV(
                clone(estimator), grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=n_jobs
            )
            search.fit(X_train, y_train, groups=None if groups is None else groups[train_idx])
            model, params = search.best_estimator_, search.best_params_
        else:
            model = clone(estimator).set_params(**fixed_params).fit(X_train, y_train)
            params = fixed_params
        y_pred = model.predict(X_test)
        folds.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **regression_metrics(y_test, y_pred),
                "params": json.dumps({k.removeprefix("model__"): v for k, v in params.items()}),
            }
        )
        preds.append(pd.DataFrame({"row": test_idx, "fold": fold, "y_true": y_test.to_numpy(), "y_pred": y_pred}))
    return folds, pd.concat(preds, ignore_index=True)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=0, as ``np.std`` in the original)
    of each metric across outer folds."""
    agg = folds.groupby(["scheme", "model"], sort=False)[["rmse", "mae", "r2"]]
    mean, std = agg.mean(), agg.std(ddof=0)
    out = pd.concat({"mean": mean, "std": std}, axis=1)
    out.columns = [f"{metric}_{stat}" for stat, metric in out.columns]
    return out[[f"{m}_{s}" for m in ("rmse", "mae", "r2") for s in ("mean", "std")]].reset_index()
