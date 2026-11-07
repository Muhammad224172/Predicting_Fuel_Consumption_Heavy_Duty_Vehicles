"""Model definitions and hyper-parameter grids."""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .selection import CorrelationSelector

# Grids copied from the original notebook (cells 103 and 122).
DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "dummy": {},
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "random_forest": {
        "n_estimators": [150, 200],
        "max_features": ["sqrt"],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5, 7],
        "min_samples_leaf": [2, 3],
    },
    "xgboost": {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [4, 5, 7],
    },
}


def _base_estimator(name: str, seed: int):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge()
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=1)
    raise ValueError(f"Unknown model {name!r}; choose from {sorted(DEFAULT_GRIDS)}")


def make_model(
    name: str, seed: int, corr_top_k: int | None = None, grid_override: dict | None = None
) -> tuple[Pipeline, dict[str, list]]:
    """Return an unfitted pipeline and its grid (keys prefixed for the pipeline).

    Every model is wrapped in a pipeline: optional correlation-based selection,
    standardisation for ridge (tree models do not need it), then the
    regressor. Because selection and scaling are pipeline steps, they are
    fitted on training folds only.
    """
    steps = []
    if corr_top_k:
        steps.append(("select", CorrelationSelector(k=corr_top_k)))
    if name == "ridge":
        steps.append(("scale", StandardScaler()))
    steps.append(("model", _base_estimator(name, seed)))
    grid = grid_override if grid_override is not None else DEFAULT_GRIDS[name]
    return Pipeline(steps), {f"model__{k}": list(v) for k, v in grid.items()}
