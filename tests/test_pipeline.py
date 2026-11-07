"""End-to-end smoke tests on synthetic data (checks plumbing, not accuracy)."""

import json

import numpy as np
import pandas as pd
import yaml

from fuelcons.cli import main
from fuelcons.config import config_from_dict, load_config
from fuelcons.pipeline import run

TINY_GRIDS = {
    "ridge": {"alpha": [0.1, 10.0]},
    "random_forest": {"n_estimators": [10], "min_samples_leaf": [2]},
    "xgboost": {"n_estimators": [20], "max_depth": [2, 3]},
}


def _config(tmp_path, raw_dir, **overrides):
    data = {
        "name": "smoke",
        "raw_dir": str(raw_dir),
        "processed_dir": str(tmp_path / "processed"),
        "results_dir": str(tmp_path / "results"),
        "n_jobs": 1,
        "window": {"unit": "rows", "size": 120, "step": 40},
        "param_grids": TINY_GRIDS,
        "cv": {"n_splits": 3, "inner_splits": 2, "schemes": ["kfold", "group_record", "group_vehicle"]},
    }
    data.update(overrides)
    return config_from_dict(data)


def test_end_to_end(tmp_path, raw_dir):
    cfg = _config(tmp_path, raw_dir)
    out = run(cfg)
    folds = pd.read_csv(out / "folds.csv")
    assert len(folds) == 3 * 4 * 3  # schemes x models x folds
    assert np.isfinite(folds[["rmse", "mae", "r2"]]).all().all()
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["model"]) == {"dummy", "ridge", "random_forest", "xgboost"}
    preds = pd.read_csv(out / "predictions.csv")
    for _, g in preds.groupby(["scheme", "model"]):
        assert g["row"].is_unique  # every window predicted exactly once per scheme/model
    info = json.loads((out / "run_info.json").read_text())
    assert info["data"]["n_vehicles"] == 4 and info["data"]["n_records"] <= 8
    assert 0 < info["data"]["n_windows"] <= info["data"]["n_windows_before_outlier_removal"]
    for name in ("cv_comparison", "pred_vs_actual_xgboost", "target_correlation", "feature_importance_xgboost"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
    assert "| group_vehicle | xgboost |" in (out / "summary.md").read_text()
    assert (tmp_path / "processed" / "windows_rows_120_40.csv").exists()  # cached for the next run


def test_original_protocol_with_correlation_filter(tmp_path, raw_dir):
    cfg = _config(
        tmp_path,
        raw_dir,
        name="orig",
        models=["xgboost"],
        features={"set": "no_median", "corr_top_k": 10},
        cv={"n_splits": 3, "schemes": ["kfold"], "tuning": "original"},
    )
    folds = pd.read_csv(run(cfg) / "folds.csv")
    assert folds["params"].nunique() == 1  # tuned once, re-used in every fold


def test_cli_and_shipped_configs(tmp_path, raw_dir):
    for path in ("configs/default.yaml", "configs/original_protocol.yaml"):
        load_config(path)  # shipped configs parse
    cfg_path = tmp_path / "cli.yaml"
    cfg = _config(
        tmp_path, raw_dir, name="cli", models=["dummy", "ridge"], window={"unit": "seconds", "size": 120, "step": 60}
    )
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()))
    main(["--config", str(cfg_path), "--n-jobs", "1"])
    assert (tmp_path / "results" / "cli" / "summary.csv").exists()
