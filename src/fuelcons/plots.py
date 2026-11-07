"""Figures generated from the saved results (no figure is edited by hand)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
# Fixed colour per model (follows the entity, not its rank).
MODEL_COLORS = {
    "dummy": "#8a8984",
    "ridge": "#2a78d6",
    "random_forest": "#eb6834",
    "xgboost": "#1baf7a",
}
MODEL_LABELS = {"dummy": "Mean baseline", "ridge": "Ridge", "random_forest": "Random forest", "xgboost": "XGBoost"}
SCHEME_LABELS = {
    "kfold": "Shuffled KFold\n(original)",
    "group_record": "GroupKFold\nby record",
    "group_vehicle": "GroupKFold\nby vehicle",
}
METRIC_LABELS = {"rmse": "RMSE (lower is better)", "mae": "MAE (lower is better)", "r2": "R² (higher is better)"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(ncols: int, width: float, height: float):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        _style(ax)
    return fig, axes[0]


def plot_cv_comparison(folds: pd.DataFrame, path: Path, metrics=("rmse", "r2")) -> None:
    """Per-fold scores (dots) and their mean (bar) for each model and CV scheme."""
    schemes = list(dict.fromkeys(folds["scheme"]))
    models = list(dict.fromkeys(folds["model"]))
    fig, axes = _figure(len(metrics), 5.2 * len(metrics), 4.2)
    width = 0.8 / len(models)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, metrics, strict=True):
        for j, model in enumerate(models):
            color = MODEL_COLORS.get(model, INK_MUTED)
            for i, scheme in enumerate(schemes):
                vals = folds.loc[(folds["scheme"] == scheme) & (folds["model"] == model), metric]
                x = i - 0.4 + width * (j + 0.5)
                ax.hlines(vals.mean(), x - width * 0.4, x + width * 0.4, color=color, linewidth=2)
                jitter = rng.uniform(-width * 0.25, width * 0.25, len(vals))
                ax.scatter(
                    x + jitter,
                    vals,
                    s=12,
                    color=color,
                    alpha=0.7,
                    linewidths=0,
                    label=MODEL_LABELS.get(model, model) if i == 0 else None,
                )
        ax.set_xticks(range(len(schemes)), [SCHEME_LABELS.get(s, s) for s in schemes])
        ax.set_title(METRIC_LABELS[metric], color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Outer-fold scores: dots are folds, lines are means", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pred_vs_actual(preds: pd.DataFrame, model: str, path: Path) -> None:
    """Out-of-fold predicted vs. observed target, one panel per CV scheme."""
    sub = preds[preds["model"] == model]
    schemes = list(dict.fromkeys(sub["scheme"]))
    fig, axes = _figure(len(schemes), 4.0 * len(schemes), 4.0)
    lo, hi = np.percentile(sub[["y_true", "y_pred"]].to_numpy(), [0.1, 99.9])
    for ax, scheme in zip(axes, schemes, strict=True):
        s = sub[sub["scheme"] == scheme]
        ax.scatter(s["y_true"], s["y_pred"], s=4, alpha=0.25, linewidths=0, color=MODEL_COLORS.get(model, INK))
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(SCHEME_LABELS.get(scheme, scheme).replace("\n", " "), color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Observed fuel per 100 km (masked units)", color=INK_MUTED)
    axes[0].set_ylabel("Predicted", color=INK_MUTED)
    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)}: out-of-fold predictions", color=INK, fontsize=12, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ranking(values: pd.Series, title: str, xlabel: str, path: Path, top: int = 15) -> None:
    """Horizontal bars for a ranked series (feature importance or |correlation|)."""
    vals = values.sort_values(ascending=False).head(top)[::-1]
    fig, axes = _figure(1, 7.0, 0.32 * len(vals) + 1.2)
    ax = axes[0]
    ax.barh(vals.index, vals.to_numpy(), color=MODEL_COLORS["ridge"], height=0.7)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
