"""Probe the gap between the report's numbers and the notebook's saved outputs.

Re-runs the original 10-fold shuffled KFold (notebook row order, 58 features)
with FIXED hyper-parameters, varying one thing at a time:

* XGBoost with n_estimators=300 (hyper-parameters printed in the report) vs.
  700 (chosen by the grid search saved in the notebook);
* random forest with the report's hyper-parameters under several seeds (the
  notebook set no random_state);
* the notebook's record order vs. numeric order (changes the fold assignment).

Usage: python scripts/discrepancy_check.py [--raw-dir PATH]
Writes results/discrepancy/discrepancy.csv and .md.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fuelcons.cleaning import remove_outliers  # noqa: E402
from fuelcons.config import load_config  # noqa: E402
from fuelcons.evaluation import cross_validate  # noqa: E402
from fuelcons.features import TARGET  # noqa: E402
from fuelcons.models import make_model  # noqa: E402
from fuelcons.pipeline import load_windows, order_records, read_record_order  # noqa: E402
from fuelcons.selection import feature_columns  # noqa: E402

XGB_REPORT = {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 300}
XGB_NOTEBOOK = {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 700}
RF_REPORT = {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 2}


def evaluate(table, model, params, seed=42, model_seed=42):
    X, y = table[feature_columns("original")], table[TARGET]
    estimator, _ = make_model(model, model_seed)
    estimator.set_params(**{f"model__{k}": v for k, v in params.items()})
    folds, _ = cross_validate(
        X, y, None, estimator, {}, "kfold", n_splits=10, inner_splits=5, tuning="original", seed=seed, n_jobs=1
    )
    df = pd.DataFrame(folds)
    return {
        **{f"{m}_mean": df[m].mean() for m in ("rmse", "mae", "r2")},
        **{f"{m}_std": df[m].std(ddof=0) for m in ("rmse", "mae", "r2")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "original_protocol.yaml")
    cfg.processed_dir = str(ROOT / cfg.processed_dir)
    raw_dir = args.raw_dir or os.environ.get("FUELCONS_RAW_DIR") or ROOT / cfg.raw_dir
    windows = load_windows(cfg, raw_dir)
    numeric, _ = remove_outliers(windows, cfg.outliers.columns, cfg.outliers.iqr_k)
    notebook = order_records(windows, read_record_order(ROOT / cfg.record_order_file))
    notebook, _ = remove_outliers(notebook, cfg.outliers.columns, cfg.outliers.iqr_k)

    runs = [
        ("xgboost", "report hyper-parameters (300 trees)", "notebook", XGB_REPORT, 42),
        ("xgboost", "notebook hyper-parameters (700 trees)", "notebook", XGB_NOTEBOOK, 42),
        ("xgboost", "notebook hyper-parameters (700 trees)", "numeric", XGB_NOTEBOOK, 42),
        *[("random_forest", f"report hyper-parameters, seed {s}", "notebook", RF_REPORT, s) for s in range(5)],
        ("random_forest", "report hyper-parameters, seed 0", "numeric", RF_REPORT, 0),
    ]
    rows = []
    for model, label, order, params, model_seed in runs:
        table = notebook if order == "notebook" else numeric
        print(f"{model}: {label}, {order} order", flush=True)
        rows.append(
            {"model": model, "setting": label, "row_order": order, **evaluate(table, model, params, 42, model_seed)}
        )
    out = ROOT / "results" / "discrepancy"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "discrepancy.csv", index=False, float_format="%.6g")
    lines = [
        "# Original protocol with fixed hyper-parameters",
        "",
        "Shuffled 10-fold KFold (seed 42), 58 original features, 10,087 windows. Mean ± std over folds.",
        "",
        "| Model | Setting | Row order | RMSE | MAE | R² |",
        "|---|---|---|---|---|---|",
    ]
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.model} | {r.setting} | {r.row_order} | {r.rmse_mean:.4f} ± {r.rmse_std:.4f} | "
            f"{r.mae_mean:.4f} ± {r.mae_std:.4f} | {r.r2_mean:.4f} ± {r.r2_std:.4f} |"
        )
    (out / "discrepancy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
