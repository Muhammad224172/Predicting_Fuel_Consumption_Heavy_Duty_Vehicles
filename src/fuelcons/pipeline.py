"""End-to-end run: raw records -> windows -> CV results, tables and figures."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone

from . import plots
from .cleaning import remove_outliers
from .config import Config
from .evaluation import cross_validate, scheme_groups, summarise
from .features import TARGET, build_window_table
from .io import list_records, record_id
from .models import make_model
from .selection import correlation_ranking, feature_columns

log = logging.getLogger("fuelcons")


def load_windows(cfg: Config, raw_dir: str | Path, rebuild: bool = False) -> pd.DataFrame:
    """Window table for the configured window settings, cached as CSV.

    The cache is reused only if it was built from the same raw folder and the
    same set of record files (checked via a small sidecar JSON file).
    """
    w = cfg.window
    cache = Path(cfg.processed_dir) / f"windows_{w.unit}_{w.size:g}_{w.step:g}.csv"
    meta_path = cache.with_suffix(".json")
    meta = {"raw_dir": str(Path(raw_dir).resolve()), "records": [record_id(p) for p in list_records(raw_dir)]}
    if cache.exists() and meta_path.exists() and not rebuild:
        if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            log.info("Loading cached window table %s", cache)
            return pd.read_csv(cache, parse_dates=["start_time", "end_time"])
    log.info("Building window table from %s (%s: size=%g, step=%g)", raw_dir, w.unit, w.size, w.step)
    table = build_window_table(raw_dir, w.unit, w.size, w.step)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return table


def read_record_order(path: str | Path) -> list[int]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [int(line) for line in lines if line.strip() and not line.lstrip().startswith("#")]


def order_records(table: pd.DataFrame, order: list[int]) -> pd.DataFrame:
    """Stack records in ``order`` (unlisted records go last, by id)."""
    rank = table["record_id"].map({rec: i for i, rec in enumerate(order)})
    key = rank.fillna(len(order) + table["record_id"])
    return (
        table.assign(_key=key)
        .sort_values(["_key", "window_index"], kind="stable")
        .drop(columns="_key")
        .reset_index(drop=True)
    )


def prepare_table(cfg: Config, raw_dir=None, rebuild: bool = False) -> tuple[pd.DataFrame, int, list[dict]]:
    """Window table in the configured record order, after outlier removal."""
    table = load_windows(cfg, raw_dir or cfg.raw_dir, rebuild)
    if cfg.record_order_file:
        table = order_records(table, read_record_order(cfg.record_order_file))
    n_before = len(table)
    table, outlier_log = remove_outliers(table, cfg.outliers.columns, cfg.outliers.iqr_k)
    return table, n_before, outlier_log


def redraw_figures(cfg: Config, raw_dir=None) -> Path:
    """Regenerate the figures of an earlier run from its saved results."""
    out = Path(cfg.results_dir) / cfg.name
    table, _, _ = prepare_table(cfg, raw_dir)
    X, y = table[feature_columns(cfg.features.set)], table[TARGET]
    folds = pd.read_csv(out / "folds.csv")
    preds = pd.read_csv(out / "predictions.csv")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _make_figures(cfg, X, y, folds, preds, out / "figures")
    return out


def run(cfg: Config, raw_dir: str | Path | None = None, rebuild: bool = False) -> Path:
    """Run every (CV scheme, model) pair in ``cfg`` and write results to
    ``<results_dir>/<cfg.name>/``. Returns that directory."""
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    table, n_windows_before, outlier_log = prepare_table(cfg, raw_dir, rebuild)
    features = feature_columns(cfg.features.set)
    X, y = table[features], table[TARGET]
    log.info("%d windows (%d before outlier removal), %d features", len(table), n_windows_before, len(features))

    all_folds, all_preds = [], []
    for scheme in cfg.cv.schemes:
        groups = scheme_groups(table, scheme)
        for name in cfg.models:
            estimator, grid = make_model(name, cfg.seed, cfg.features.corr_top_k, cfg.param_grids.get(name))
            log.info("CV scheme=%s model=%s tuning=%s", scheme, name, cfg.cv.tuning)
            folds, preds = cross_validate(
                X,
                y,
                groups,
                estimator,
                grid,
                scheme,
                n_splits=cfg.cv.n_splits,
                inner_splits=cfg.cv.inner_splits,
                tuning=cfg.cv.tuning,
                seed=cfg.seed,
                n_jobs=cfg.n_jobs,
            )
            all_folds += [{"scheme": scheme, "model": name, **f} for f in folds]
            preds.insert(0, "model", name)
            preds.insert(0, "scheme", scheme)
            preds["record_id"] = table["record_id"].to_numpy()[preds["row"]]
            preds["vehicle_id"] = table["vehicle_id"].to_numpy()[preds["row"]]
            all_preds.append(preds)

    folds_df = pd.DataFrame(all_folds)
    preds_df = pd.concat(all_preds, ignore_index=True)
    summary = summarise(folds_df)

    out = Path(cfg.results_dir) / cfg.name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(out / "folds.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    preds_df.to_csv(out / "predictions.csv", index=False, float_format="%.6g")

    info = {
        "config": cfg.to_dict(),
        "started_utc": started.isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "git_commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "data": {
            "n_records": int(table["record_id"].nunique()),
            "n_vehicles": int(table["vehicle_id"].nunique()),
            "n_windows_before_outlier_removal": n_windows_before,
            "n_windows": len(table),
            "outlier_removal": outlier_log,
            "n_features": len(features),
            "target_mean": float(y.mean()),
            "target_std": float(y.std(ddof=0)),
        },
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text(_summary_markdown(cfg, summary, info), encoding="utf-8")
    _make_figures(cfg, X, y, folds_df, preds_df, out / "figures")
    log.info("Results written to %s", out)
    return out


def _make_figures(cfg, X, y, folds_df, preds_df, fig_dir: Path) -> None:
    plots.plot_cv_comparison(folds_df, fig_dir / "cv_comparison.png")
    model = "xgboost" if "xgboost" in cfg.models else cfg.models[-1]
    plots.plot_pred_vs_actual(preds_df, model, fig_dir / f"pred_vs_actual_{model}.png")
    plots.plot_ranking(
        correlation_ranking(X, y),
        "Absolute Pearson correlation with the target (all windows)",
        "|r|",
        fig_dir / "target_correlation.png",
        top=20,
    )
    if "xgboost" in cfg.models:
        importance = _xgb_importance(cfg, X, y, folds_df)
        plots.plot_ranking(
            importance,
            "XGBoost feature importance (gain), refit on all windows",
            "Share of total gain",
            fig_dir / "feature_importance_xgboost.png",
        )


def _xgb_importance(cfg: Config, X: pd.DataFrame, y: pd.Series, folds_df: pd.DataFrame) -> pd.Series:
    """Refit XGBoost on all windows with its most frequently selected
    parameters (first CV scheme) and return normalised gain importances.
    Descriptive only; not used for any reported score."""
    rows = folds_df[(folds_df["model"] == "xgboost") & (folds_df["scheme"] == cfg.cv.schemes[0])]
    params = json.loads(Counter(rows["params"]).most_common(1)[0][0])
    estimator, _ = make_model("xgboost", cfg.seed, cfg.features.corr_top_k, {})
    model = clone(estimator).set_params(**{f"model__{k}": v for k, v in params.items()}).fit(X, y)
    names = np.asarray(X.columns)
    if "select" in model.named_steps:
        names = names[model.named_steps["select"].support_]
    gain = model.named_steps["model"].feature_importances_
    return pd.Series(gain / gain.sum(), index=names)


def _summary_markdown(cfg: Config, summary: pd.DataFrame, info: dict) -> str:
    d = info["data"]
    w = cfg.window
    lines = [
        f"# Results: `{cfg.name}`",
        "",
        f"- Run started (UTC): {info['started_utc']}; git commit: `{info['git_commit']}`",
        f"- Data: {d['n_records']} records, {d['n_vehicles']} vehicles, "
        f"{d['n_windows']} windows after outlier removal ({d['n_windows_before_outlier_removal']} before)",
        f"- Windows: {w.size:g} {w.unit}, step {w.step:g} {w.unit}; feature set `{cfg.features.set}` "
        f"({d['n_features']} features), correlation filter k={cfg.features.corr_top_k}",
        f"- CV: {cfg.cv.n_splits} outer folds, tuning `{cfg.cv.tuning}`"
        + (f" ({cfg.cv.inner_splits} inner folds)" if cfg.cv.tuning == "nested" else ""),
        f"- Target `{TARGET}`: mean {d['target_mean']:.4f}, std {d['target_std']:.4f} (masked fuel units per 100 km)",
        "",
        "Mean ± standard deviation over outer folds.",
        "",
        "| CV scheme | Model | RMSE | MAE | R² |",
        "|---|---|---|---|---|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.scheme} | {r.model} | {r.rmse_mean:.4f} ± {r.rmse_std:.4f} | "
            f"{r.mae_mean:.4f} ± {r.mae_std:.4f} | {r.r2_mean:.4f} ± {r.r2_std:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
