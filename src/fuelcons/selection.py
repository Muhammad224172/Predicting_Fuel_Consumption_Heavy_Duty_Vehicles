"""Feature sets and correlation-based feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .features import AGG_SIGNALS

NO_MEDIAN_STATS = ("mean", "max", "min", "std", "range")
# Signals whose window statistics directly encode the target
# (fuel_per_100km = fuel range / distance range). Only the distance range is
# kept as a feature, as in the original.
TARGET_SIGNALS = ("EngTotalFuelUsed", "HghRslutionTotalVehicleDistance")
_MODEL_SIGNALS = [s for s in AGG_SIGNALS if s not in TARGET_SIGNALS]

# The 58 columns of ``mean_features_columns`` in the original notebook (cell 99).
# ``slope_positive`` / ``slope_negative`` there were point values at the first
# row of each window; slope_positive window statistics were never computed.
ORIGINAL_FEATURES = [
    "slope_positive_start",
    "slope_negative_start",
    *[f"slope_negative_{s}" for s in NO_MEDIAN_STATS],
    "HghRslutionTotalVehicleDistance_range",
    *[
        f"{sig}_{s}"
        for sig in _MODEL_SIGNALS
        if sig not in ("slope_positive", "slope_negative")
        for s in NO_MEDIAN_STATS
    ],
]


def feature_columns(feature_set: str) -> list[str]:
    """Column names for a named feature set.

    * ``original``: exactly the 58 features of the original final models.
    * ``no_median``: the same idea without the notebook's quirks: every
      non-median window statistic of every non-target signal, including
      ``slope_positive``, plus the window distance.
    * ``all``: as ``no_median`` but with medians.
    """
    if feature_set == "original":
        return list(ORIGINAL_FEATURES)
    if feature_set in ("no_median", "all"):
        stats = NO_MEDIAN_STATS if feature_set == "no_median" else (*NO_MEDIAN_STATS, "median")
        return ["HghRslutionTotalVehicleDistance_range"] + [f"{sig}_{s}" for sig in _MODEL_SIGNALS for s in stats]
    raise ValueError(f"Unknown feature set {feature_set!r}")


def correlation_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Absolute Pearson correlation of each column with ``y``, descending."""
    corr = X.apply(lambda col: _abs_corr(col.to_numpy(float), y.to_numpy(float)))
    return corr.sort_values(ascending=False)


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Keep the ``k`` features with the highest |Pearson r| to the target.

    Used inside a Pipeline, so the ranking is learned on training folds only.
    """

    def __init__(self, k: int = 30):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        scores = np.array([_abs_corr(X[:, j], y) for j in range(X.shape[1])])
        self.scores_ = scores
        self.support_ = np.sort(np.argsort(-scores, kind="stable")[: self.k])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


def _abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))
