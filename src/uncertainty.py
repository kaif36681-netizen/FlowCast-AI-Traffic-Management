"""Prediction intervals for the volume forecast (FR-11).

The PRD asks for a confidence estimate on every prediction. This is the part of
the build where it is easiest to produce something decorative — a fixed ±10%
band, or the standard deviation of the training residuals reused for every row —
so the approach is stated explicitly and then measured.

Two independent estimates are produced:

    quantile gradient boosting   three XGBoost models fitted with the pinball
                                 loss at the 10th, 50th and 90th percentiles.
                                 The band is genuinely conditional: it widens on
                                 peak windows and in rain, and narrows overnight.

    MC-dropout spread            produced in dl_model.py by sampling the LSTM
                                 with dropout active.

The quantile band is what the dashboard shows, because it is the one that can be
checked: empirical coverage on the test window is measured below and reported.
A nominal 80% interval that actually covers 62% of outcomes is worse than no
interval at all, and the number is printed either way.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from ml_models import load_splits
from utils import get_logger, load_config, save_json, set_seed

QUANTILES = [0.10, 0.50, 0.90]


def fit_quantile_models(Xtr, ytr, cfg, log):
    """Fit one booster per quantile using the pinball (quantile) objective."""
    seed = cfg["project"]["random_seed"]
    models = {}
    for q in QUANTILES:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            tree_method="hist", n_jobs=-1, random_state=seed,
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
        )
        model.fit(Xtr, ytr, verbose=False)
        models[q] = model
        log.info("  fitted quantile model for q=%.2f", q)
    return models


def coverage_report(y_true, lower, upper, log) -> dict:
    """Measure what the nominal 80% interval actually covers."""
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    out = {
        "nominal_coverage": 0.80,
        "empirical_coverage": float(inside.mean()),
        "mean_interval_width": float(width.mean()),
        "median_interval_width": float(np.median(width)),
        "width_as_pct_of_mean_volume": float(width.mean() / y_true.mean() * 100),
    }
    log.info("interval coverage: nominal 80%%, empirical %.1f%% "
             "(mean width %.0f vehicles, %.0f%% of mean volume)",
             100 * out["empirical_coverage"], out["mean_interval_width"],
             out["width_as_pct_of_mean_volume"])
    return out


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("uncertainty", cfg)
    set_seed(cfg["project"]["random_seed"])
    log.info("=== Prediction intervals (FR-11) ===")

    _, features, train, val, test, _, _ = load_splits(cfg, log)
    Xtr = train[features].values.astype("float32")
    Xva = val[features].values.astype("float32")
    Xte = test[features].values.astype("float32")
    ytr, yte = train["y_volume"].values, test["y_volume"].values

    models = fit_quantile_models(Xtr, ytr, cfg, log)

    lower = models[0.10].predict(Xte)
    median = models[0.50].predict(Xte)
    upper = models[0.90].predict(Xte)
    # Quantile crossing: separately fitted quantiles are not guaranteed to be
    # ordered. Sorting the three predictions per row enforces monotonicity
    # without retraining under a joint constraint.
    stacked = np.sort(np.vstack([lower, median, upper]), axis=0)
    lower, median, upper = stacked[0], stacked[1], stacked[2]

    cov = coverage_report(yte, lower, upper, log)

    # Where is the model least sure? Breaking width down by condition tells an
    # operator when to trust the point forecast and when to hedge.
    frame = test[["road_id", "timestamp", "is_peak", "rain_flag"]].copy()
    frame["width"] = upper - lower
    by_condition = (frame.assign(
        condition=np.select(
            [frame.is_peak.eq(1) & frame.rain_flag.eq(1),
             frame.is_peak.eq(1),
             frame.rain_flag.eq(1)],
            ["Peak + rain", "Peak", "Rain"], default="Off-peak, dry"))
        .groupby("condition")["width"].agg(["mean", "size"]))
    log.info("interval width by condition:\n%s", by_condition.round(1).to_string())

    joblib.dump(models, Path(cfg["paths"]["models"]) / "quantile_volume.joblib")

    intervals = test[["road_id", "timestamp"]].copy()
    intervals["volume_p10"] = lower
    intervals["volume_p50"] = median
    intervals["volume_p90"] = upper
    intervals.to_parquet(
        Path(cfg["paths"]["processed"]) / "volume_intervals.parquet", index=False)

    out = {"coverage": cov,
           "width_by_condition": by_condition.round(2).to_dict(orient="index")}
    save_json(out, Path(cfg["paths"]["reports"]) / "uncertainty.json")
    log.info("intervals complete")
    return out


if __name__ == "__main__":
    run()
