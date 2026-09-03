"""M3 — Feature Engineering.

Builds the supervised learning table. One row = one (segment, target window T).

The single most important decision in this module is the *framing*:

    targets  are measured at window T
    features are measured at window T-1 or earlier, plus attributes of T that are
             genuinely known in advance (clock time, day of week, holiday and
             event calendar, the published signal plan, road capacity)

That framing is what makes the problem real. Traffic volume at T is trivially
predictable from occupancy at T, and congestion_level at T is a deterministic
banding of volume at T — a model given either would score near-perfectly and
forecast nothing. Every dynamic feature here is therefore shifted by at least
one window within its own segment.

Weather is used at T-1 rather than T. In production an operator would have a
short-range forecast for T, so this is the conservative choice; because weather
is broadcast hourly, T-1 shares the same observation as T half the time anyway.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_logger, load_config, read_table, save_json, save_parquet

# Columns measured at T. They are targets or target-derived; none may be a feature.
TARGET_SOURCE_COLS = [
    "traffic_volume", "avg_speed", "occupancy", "congestion_level",
    "travel_time", "accident_count", "vc_ratio",
]

CONGESTION_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]

DYNAMIC_COLS = ["traffic_volume", "avg_speed", "occupancy", "vc_ratio"]

WEATHER_COLS = ["temperature", "rainfall", "visibility"]


def add_targets(df: pd.DataFrame, log) -> pd.DataFrame:
    """Attach the four modelling targets, all measured at window T."""
    df["y_volume"] = df["traffic_volume"].astype("float32")
    df["y_speed"] = df["avg_speed"].astype("float32")
    df["y_travel_time"] = df["travel_time"].astype("float32")
    df["y_congestion"] = pd.Categorical(df["congestion_level"],
                                        categories=CONGESTION_ORDER, ordered=True)
    df["y_congestion_code"] = df["y_congestion"].cat.codes.astype("int8")
    # Accident risk is framed as "any reported incident in this window", which is
    # what the operations team acts on. Counts above 1 are too rare to model.
    df["y_accident"] = (df["accident_count"] > 0).astype("int8")
    log.info("targets: volume/speed/travel_time (regression), congestion (4-class), "
             "accident (binary, base rate %.3f%%)", 100 * df["y_accident"].mean())
    return df


def add_time_features(df: pd.DataFrame, log) -> pd.DataFrame:
    """Clock and calendar features of the *target* window — known in advance."""
    ts = df["timestamp"]
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int16")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["day_of_month"] = ts.dt.day.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["week_of_year"] = ts.dt.isocalendar().week.astype("int16")

    # Cyclical encodings: midnight and 23:30 are adjacent in time, and a raw
    # hour column tells a linear model they are 23 units apart.
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")

    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_am_peak"] = hour_frac.between(7, 10.5, inclusive="left").astype("int8")
    df["is_pm_peak"] = hour_frac.between(16.5, 20, inclusive="left").astype("int8")
    df["is_peak"] = (df["is_am_peak"] | df["is_pm_peak"]).astype("int8")
    df["is_night"] = ((hour_frac < 6) | (hour_frac >= 22)).astype("int8")

    # Interactions the PRD calls for: a holiday suppresses the peak, an event
    # amplifies it, and roadworks bite hardest when the corridor is already busy.
    df["holiday_x_peak"] = (df["public_holiday"] * df["is_peak"]).astype("int8")
    df["event_x_peak"] = (df["event_flag"] * df["is_peak"]).astype("int8")
    df["roadwork_x_peak"] = (df["roadwork_flag"] * df["is_peak"]).astype("int8")
    log.info("time features: cyclical encodings, peak flags and calendar interactions")
    return df


def add_lag_features(df: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    """Per-segment lags of the dynamic sensor channels.

    grouped by road_id so a lag never reaches across a segment boundary, and
    computed on the rebuilt uniform grid so lag_1 is always exactly 30 minutes
    back rather than 'the previous row that happened to exist'.
    """
    lags = cfg["features"]["lags"]
    grp = df.groupby("road_id", observed=True)
    new = {}
    for col in DYNAMIC_COLS:
        for lag in lags:
            new[f"{col}_lag{lag}"] = grp[col].shift(lag).astype("float32")
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # Momentum: how fast conditions are changing into the prediction point.
    df["volume_delta_1"] = (df["traffic_volume_lag1"] - df["traffic_volume_lag2"]).astype("float32")
    df["volume_delta_2"] = (df["traffic_volume_lag2"] - df["traffic_volume_lag3"]).astype("float32")
    df["speed_delta_1"] = (df["avg_speed_lag1"] - df["avg_speed_lag2"]).astype("float32")
    # Same window yesterday vs. the day before — is today running above trend?
    df["volume_dod_change"] = (df["traffic_volume_lag48"] -
                               df["traffic_volume_lag336"]).astype("float32")
    log.info("lag features: %d lags x %d channels", len(lags), len(DYNAMIC_COLS))
    return df


def add_rolling_features(df: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    """Rolling mean and standard deviation over the windows *before* T.

    The shift(1) before rolling is the leakage guard: without it the window
    closes on T itself and the mean contains the answer.
    """
    windows = cfg["features"]["rolling_windows"]
    grp = df.groupby("road_id", observed=True)
    new = {}
    for col in ["traffic_volume", "avg_speed"]:
        shifted = grp[col].shift(1)
        for w in windows:
            roll = shifted.groupby(df["road_id"], observed=True).rolling(
                w, min_periods=max(2, w // 2))
            new[f"{col}_rollmean{w}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
            new[f"{col}_rollstd{w}"] = roll.std().reset_index(level=0, drop=True).astype("float32")
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # Deviation of the most recent reading from its own short-run baseline —
    # this is what "something unusual is starting" looks like numerically.
    df["volume_vs_roll8"] = (df["traffic_volume_lag1"] -
                             df["traffic_volume_rollmean8"]).astype("float32")
    df["speed_vs_roll8"] = (df["avg_speed_lag1"] -
                            df["avg_speed_rollmean8"]).astype("float32")
    log.info("rolling features: mean/std over %s windows for volume and speed", windows)
    return df


def add_capacity_and_weather_features(df: pd.DataFrame, log) -> pd.DataFrame:
    """Capacity headroom and weather condition, both lagged where dynamic."""
    grp = df.groupby("road_id", observed=True)

    # Capacity is static; the ratio uses lagged volume so it stays honest.
    cap_30 = df["road_capacity"] / 2.0
    df["capacity_30min"] = cap_30.astype("float32")
    df["vc_lag1"] = (df["traffic_volume_lag1"] / cap_30).astype("float32")
    df["headroom_lag1"] = (cap_30 - df["traffic_volume_lag1"]).astype("float32")
    df["signal_capacity_ratio"] = (df["signal_timing"] / df["road_capacity"] * 1000).astype("float32")

    for col in WEATHER_COLS:
        df[f"{col}_lag1"] = grp[col].shift(1).astype("float32")
    df["weather_condition_lag1"] = grp["weather_condition"].shift(1)
    df["rain_flag"] = (df["rainfall_lag1"] > 0.1).astype("int8")
    df["heavy_rain_flag"] = (df["rainfall_lag1"] > 2.5).astype("int8")
    df["low_visibility_flag"] = (df["visibility_lag1"] < 2000).astype("int8")
    df["temp_band"] = pd.cut(df["temperature_lag1"],
                             bins=[-np.inf, 10, 20, 30, np.inf],
                             labels=["cold", "mild", "warm", "hot"]).astype(object)
    # Rain during a peak is the operationally interesting case, not rain as such.
    df["rain_x_peak"] = (df["rain_flag"] * df["is_peak"]).astype("int8")
    df["rain_3h"] = (df.groupby("road_id", observed=True)["rainfall"]
                       .shift(1).groupby(df["road_id"], observed=True)
                       .rolling(6, min_periods=1).sum()
                       .reset_index(level=0, drop=True).astype("float32"))
    log.info("capacity + weather features: headroom, V/C, rain and visibility flags")
    return df


def encode_categoricals(df: pd.DataFrame, log) -> pd.DataFrame:
    """One-hot the nominal features; congestion history is encoded ordinally."""
    nominal = ["weather_condition_lag1", "temp_band"]
    dummies = pd.get_dummies(df[nominal].astype(str), prefix=nominal,
                             dummy_na=False, dtype="int8")
    df = pd.concat([df, dummies], axis=1)

    # Congestion is ordered, so its lagged history is a single ordinal column
    # rather than four indicator columns.
    order_map = {c: i for i, c in enumerate(CONGESTION_ORDER)}
    lagged = df.groupby("road_id", observed=True)["congestion_level"].shift(1)
    df["congestion_lag1_ord"] = lagged.map(order_map).astype("float32")
    lagged48 = df.groupby("road_id", observed=True)["congestion_level"].shift(48)
    df["congestion_lag48_ord"] = lagged48.map(order_map).astype("float32")

    # Segment identity as an integer code — tree models split on it happily and
    # it lets one global model learn per-segment offsets.
    df["road_code"] = pd.Categorical(df["road_id"]).codes.astype("int16")
    log.info("encoding: %d one-hot columns, ordinal congestion history, segment code",
             dummies.shape[1])
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the modelling feature list, with the leakage exclusions applied."""
    exclude = set(TARGET_SOURCE_COLS) | {
        "road_id", "road_name", "timestamp", "date", "weather_station_id",
        "weather_condition", "weather_condition_lag1", "temp_band",
        "holiday_name", "event_name", "congestion_derived",
        "y_volume", "y_speed", "y_travel_time", "y_congestion",
        "y_congestion_code", "y_accident",
        "temperature", "rainfall", "visibility",   # unlagged weather at T
        "tod", "dow", "segment_length_km",
    }
    feats = [c for c in df.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(feats)


def run(cfg: dict | None = None) -> dict:
    """Execute M3 and persist the analysis-ready dataset."""
    cfg = cfg or load_config()
    log = get_logger("features", cfg)
    log.info("=== M3 Feature Engineering ===")

    df = read_table(Path(cfg["paths"]["interim"]) / "merged_clean.parquet")
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    df = add_targets(df, log)
    df = add_time_features(df, log)
    df = add_lag_features(df, cfg, log)
    df = add_rolling_features(df, cfg, log)
    df = add_capacity_and_weather_features(df, log)
    df = encode_categoricals(df, log)

    features = select_feature_columns(df)

    # The deepest lag is one week; rows before that have no history and are
    # dropped rather than imputed, which would fabricate a past.
    warmup = max(cfg["features"]["lags"])
    before = len(df)
    df = df.dropna(subset=[f"traffic_volume_lag{warmup}"]).reset_index(drop=True)
    log.info("warm-up: dropped %d rows lacking a full %d-window history",
             before - len(df), warmup)

    # Any residual NaN in a feature column would break the linear and neural
    # models silently; fill with the column median and assert the result.
    med = df[features].median(numeric_only=True)
    df[features] = df[features].fillna(med)
    assert df[features].isna().sum().sum() == 0, "NaNs remain in feature matrix"

    # Two column groups are persisted here that are deliberately NOT in
    # `features`: the contemporaneous sensor channels and the unlagged weather.
    # The tabular models never see them — `feature_columns.json` is the contract
    # and they are absent from it. The LSTM needs them because its input is a
    # window of raw measurements at steps t-N..t-1, all strictly in the past of
    # the target window; excluding them would leave the sequence model with
    # nothing to read. Keeping the columns and controlling access through the
    # feature list is safer than rebuilding them in M6 from the lag columns.
    keep = (["road_id", "road_name", "timestamp", "latitude", "longitude",
             "congestion_derived", "weather_condition", "holiday_name", "event_name",
             "y_volume", "y_speed", "y_travel_time", "y_congestion",
             "y_congestion_code", "y_accident", "road_capacity",
             "segment_length_km", "traffic_volume", "avg_speed", "occupancy",
             "travel_time", "accident_count", "congestion_level",
             "vc_ratio", "temperature", "rainfall", "visibility"]
            + features)
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    out = df[keep]

    save_parquet(out, Path(cfg["paths"]["processed"]) / "flowcast_dataset.parquet")
    save_json(features, Path(cfg["paths"]["processed"]) / "feature_columns.json")

    stats = {
        "rows": int(len(out)),
        "n_features": len(features),
        "feature_columns": features,
        "warmup_rows_dropped": int(before - len(out)),
        "date_min": str(out.timestamp.min()),
        "date_max": str(out.timestamp.max()),
        "accident_base_rate": float(out.y_accident.mean()),
        "congestion_distribution": out.y_congestion.value_counts().to_dict(),
    }
    save_json(stats, Path(cfg["paths"]["reports"]) / "feature_stats.json")
    log.info("M3 complete — %d rows x %d features ready for modelling",
             len(out), len(features))
    return stats


if __name__ == "__main__":
    run()
