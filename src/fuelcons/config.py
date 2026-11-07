"""Experiment configuration (YAML file -> dataclasses)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class WindowConfig:
    unit: str = "rows"  # "rows" (original) or "seconds"
    size: float = 600
    step: float = 120


@dataclass
class OutlierConfig:
    iqr_k: float = 1.5
    # Applied one after another, in this order (as in the original notebook).
    columns: list[str] = field(default_factory=lambda: ["fuel_per_weight", "fuel_per_100km", "EngTotalFuelUsed_range"])


@dataclass
class FeatureConfig:
    set: str = "original"  # original | no_median | all
    corr_top_k: int | None = None  # optional correlation filter, fitted inside each training fold


@dataclass
class CVConfig:
    n_splits: int = 10
    inner_splits: int = 5
    schemes: list[str] = field(default_factory=lambda: ["kfold", "group_record", "group_vehicle"])
    # "nested": hyper-parameters tuned inside every outer training fold.
    # "original": tuned once on the first 80% of rows, then re-used in every fold.
    tuning: str = "nested"


@dataclass
class Config:
    name: str = "default"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    seed: int = 42
    n_jobs: int = 4
    window: WindowConfig = field(default_factory=WindowConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: list[str] = field(default_factory=lambda: ["dummy", "ridge", "random_forest", "xgboost"])
    # Optional overrides of the default hyper-parameter grids in models.py.
    param_grids: dict[str, dict[str, list]] = field(default_factory=dict)
    # Optional text file listing record ids (one per line) in the order the
    # windows should be stacked. Only the original protocol depends on row
    # order (its unshuffled 80% tuning split and the KFold assignment).
    record_order_file: str | None = None
    cv: CVConfig = field(default_factory=CVConfig)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {"window": WindowConfig, "outliers": OutlierConfig, "features": FeatureConfig, "cv": CVConfig}


def config_from_dict(data: dict) -> Config:
    data = dict(data)
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key, cls in _SECTIONS.items():
        if key in data:
            section = data[key] or {}
            bad = set(section) - {f.name for f in fields(cls)}
            if bad:
                raise ValueError(f"Unknown keys in '{key}': {sorted(bad)}")
            data[key] = cls(**section)
    return Config(**data)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("name", Path(path).stem)
    return config_from_dict(data)
