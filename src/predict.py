"""Inference API — loads persisted models and serves forecasts.

The dashboard imports this module. It never retrains: models are loaded once,
cached, and applied. That is what keeps a page interaction under a second
instead of under half an hour, and it is why FR-12 (persist and reload without
retraining) is a Must in the PRD rather than a nicety.

Batch inference over the full 25-segment corridor for one horizon is timed by
`benchmark_inference()` against the ≤ 30 s non-functional requirement.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import CONGESTION_ORDER
from utils import load_config, load_json, read_table


@lru_cache(maxsize=1)
def load_artifacts(config_path: str | None = None) -> dict:
    """Load every persisted model and lookup table exactly once."""
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])

    artifacts = {
        "cfg": cfg,
        "features": load_json(models_dir / "feature_columns.json"),
        "winners": load_json(models_dir / "winners.json"),
        "model_cards": load_json(models_dir / "model_cards.json"),
        "scaler": joblib.load(models_dir / "scaler.joblib"),
    }
    for target in ["y_volume", "y_speed", "y_travel_time", "y_congestion",
                   "y_accident"]:
        path = models_dir / f"best_{target}.joblib"
        artifacts[target] = joblib.load(path) if path.exists() else None

    q_path = models_dir / "quantile_volume.joblib"
    artifacts["quantile_volume"] = joblib.load(q_path) if q_path.exists() else None
    return artifacts


def _needs_scaling(model) -> bool:
    """Linear and margin-based models were fitted on scaled features; trees were not."""
    return model.__class__.__name__ in {
        "LinearRegression", "LogisticRegression", "SVC", "LinearSVC", "Ridge"}


def predict_frame(df: pd.DataFrame, artifacts: dict | None = None) -> pd.DataFrame:
    """Run every model over a feature frame and return one tidy prediction table."""
    art = artifacts or load_artifacts()
    features = art["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature frame is missing {len(missing)} columns, "
                         f"first few: {missing[:5]}")

    X = df[features].values.astype("float32")
    Xs = art["scaler"].transform(X).astype("float32")

    out = df[["road_id", "road_name", "timestamp", "latitude", "longitude",
              "road_capacity"]].copy() if "road_name" in df.columns else \
        df[["road_id", "timestamp"]].copy()

    for target in ["y_volume", "y_speed", "y_travel_time"]:
        model = art.get(target)
        if model is None:
            continue
        out[f"pred_{target}"] = model.predict(Xs if _needs_scaling(model) else X)

    cong = art.get("y_congestion")
    if cong is not None:
        codes = cong.predict(Xs if _needs_scaling(cong) else X)
        out["pred_congestion_code"] = codes
        out["pred_congestion"] = [CONGESTION_ORDER[int(c)] for c in codes]

    acc = art.get("y_accident")
    if acc is not None:
        Xa = Xs if _needs_scaling(acc) else X
        if hasattr(acc, "predict_proba"):
            out["pred_accident_prob"] = acc.predict_proba(Xa)[:, 1]
        else:
            score = acc.decision_function(Xa)
            out["pred_accident_prob"] = 1 / (1 + np.exp(-score))

    if art.get("quantile_volume"):
        q = art["quantile_volume"]
        stacked = np.sort(np.vstack([q[0.10].predict(X), q[0.50].predict(X),
                                     q[0.90].predict(X)]), axis=0)
        out["volume_p10"], out["volume_p50"], out["volume_p90"] = stacked
        out["confidence_width"] = stacked[2] - stacked[0]
        # A compact, operator-legible confidence label. The thresholds are
        # relative to the forecast itself, so a wide band on a busy segment is
        # not penalised the same way as a wide band on a quiet one.
        rel = out["confidence_width"] / out["pred_y_volume"].clip(lower=1)
        out["confidence"] = pd.cut(rel, [-np.inf, 0.25, 0.55, np.inf],
                                   labels=["High", "Medium", "Low"]).astype(str)

    return out


def rank_risk(predictions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Ranked accident-risk watchlist — the incident coordinator's primary view."""
    cols = ["road_id", "road_name", "timestamp", "pred_accident_prob",
            "pred_congestion", "pred_y_volume"]
    cols = [c for c in cols if c in predictions.columns]
    return (predictions[cols]
            .sort_values("pred_accident_prob", ascending=False)
            .head(top_n)
            .reset_index(drop=True))


def benchmark_inference(cfg: dict | None = None) -> dict:
    """Time batch inference over one corridor-wide horizon (NFR: ≤ 30 s)."""
    cfg = cfg or load_config()
    art = load_artifacts()
    df = read_table(Path(cfg["paths"]["processed"]) / "flowcast_dataset.parquet")
    horizon = df[df["timestamp"] == df["timestamp"].max()]

    t0 = time.time()
    _ = predict_frame(horizon, art)
    single = time.time() - t0

    day = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=1)]
    t0 = time.time()
    _ = predict_frame(day, art)
    full_day = time.time() - t0

    return {
        "segments": int(horizon["road_id"].nunique()),
        "one_horizon_seconds": round(single, 3),
        "full_day_rows": int(len(day)),
        "full_day_seconds": round(full_day, 3),
        "nfr_target_seconds": 30,
        "meets_nfr": bool(single <= 30),
    }


if __name__ == "__main__":
    print(benchmark_inference())
