"""M2 — Cleaning, Wrangling & Merge.

Order matters and is fixed:

    1. deduplicate            (before anything that averages or fills)
    2. build segment master   (static attributes survive the grid rebuild)
    3. rebuild the full grid  (absent windows become explicit NaN rows)
    4. statistical outliers   (per-segment z-score, after hard bounds in M1)
    5. impute                 (short gaps by time interpolation, long by season median)
    6. harmonise weather      (label vocabulary + hourly -> 30-minute broadcast)
    7. merge                  (weather on station+hour, calendar on date)
    8. derive congestion      (V/C banding fills the ~15% blanks)
    9. parse nested JSON      (vehicle_type_dist -> four share columns)

Every stage appends a counter to the stats dict, which becomes the
data-quality report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_logger, load_config, read_table, save_json, save_parquet

SENSOR_COLS = ["traffic_volume", "avg_speed", "occupancy"]
STATIC_COLS = ["road_name", "latitude", "longitude", "weather_station_id",
               "road_capacity"]


# --------------------------------------------------------------------------- #
# 1. Deduplication
# --------------------------------------------------------------------------- #

def deduplicate(df: pd.DataFrame, stats: dict, log) -> pd.DataFrame:
    """Drop key-level duplicates, keeping the most complete record.

    Detector retries emit the same road_id/timestamp twice. Sorting by a
    completeness score before drop_duplicates means the survivor is the row with
    the fewest nulls, not an arbitrary one.
    """
    before = len(df)
    exact = int(df.duplicated().sum())

    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = (df.sort_values(["road_id", "timestamp", "_completeness"],
                         ascending=[True, True, False])
            .drop_duplicates(subset=["road_id", "timestamp"], keep="first")
            .drop(columns="_completeness"))

    stats["duplicates_exact"] = exact
    stats["duplicates_removed"] = before - len(df)
    log.info("dedup: removed %d duplicate rows (%d were byte-identical)",
             before - len(df), exact)
    return df


# --------------------------------------------------------------------------- #
# 2-3. Segment master + complete grid
# --------------------------------------------------------------------------- #

def build_segment_master(df: pd.DataFrame, log) -> pd.DataFrame:
    """One row per segment holding its static attributes.

    Static fields are denormalised onto every reading in the source file, so the
    mode over all readings is the reliable value even where individual rows are
    missing or the segment was quarantined for that window.
    """
    agg = {c: (lambda s: s.dropna().mode().iloc[0] if s.dropna().size else np.nan)
           for c in STATIC_COLS}
    master = df.groupby("road_id").agg(agg).reset_index()
    log.info("segment master: %d segments with static attributes", len(master))
    return master


def rebuild_grid(df: pd.DataFrame, master: pd.DataFrame, cfg: dict,
                 stats: dict, log) -> pd.DataFrame:
    """Reindex onto the complete segment x 30-minute grid.

    Sensor dropouts show up in the raw file as *absent rows*, not as null values.
    Left as-is, a lag feature would silently reach across a gap and compare
    07:00 with 09:30 as if they were adjacent. Materialising the full grid makes
    every gap explicit and keeps the time axis uniform.
    """
    start, end = df["timestamp"].min(), df["timestamp"].max()
    # start/end are already tz-aware from M1; passing tz= again would double-apply it.
    full_index = pd.date_range(start, end,
                               freq=f"{cfg['data']['window_minutes']}min")
    grid = pd.MultiIndex.from_product(
        [sorted(master["road_id"].unique()), full_index],
        names=["road_id", "timestamp"],
    ).to_frame(index=False)

    before = len(df)
    df = grid.merge(df.drop(columns=STATIC_COLS, errors="ignore"),
                    on=["road_id", "timestamp"], how="left")
    df = df.merge(master, on="road_id", how="left")

    stats["grid_expected_rows"] = int(len(grid))
    stats["grid_missing_windows"] = int(len(grid) - before)
    log.info("grid: expected %d rows, %d windows were absent from the feed (%.2f%%)",
             len(grid), len(grid) - before, 100 * (len(grid) - before) / len(grid))
    return df


# --------------------------------------------------------------------------- #
# 4. Statistical outliers
# --------------------------------------------------------------------------- #

def flag_statistical_outliers(df: pd.DataFrame, cfg: dict, stats: dict,
                              log) -> pd.DataFrame:
    """Null out per-segment statistical extremes so imputation can replace them.

    M1 already removed physically impossible values by rule. What remains here
    are values that are possible but wildly atypical for that segment — a
    detector drifting rather than failing outright. They are set to NaN, never
    clipped, because clipping would invent a plausible-looking reading.
    """
    z_thresh = cfg["validation"]["z_threshold"]
    counts = {}
    for col in SENSOR_COLS:
        grp = df.groupby("road_id")[col]
        mu, sigma = grp.transform("mean"), grp.transform("std")
        z = (df[col] - mu) / sigma.replace(0, np.nan)
        extreme = z.abs() > z_thresh
        counts[col] = int(extreme.sum())
        df.loc[extreme, col] = np.nan
    stats["statistical_outliers_nulled"] = counts
    log.info("outliers: nulled %s by per-segment z-score (|z| > %.1f)",
             counts, z_thresh)
    return df


# --------------------------------------------------------------------------- #
# 5. Imputation
# --------------------------------------------------------------------------- #

def impute_sensors(df: pd.DataFrame, cfg: dict, stats: dict, log) -> pd.DataFrame:
    """Two-tier fill: interpolate short gaps, use seasonal medians for long ones.

    A one- or two-window dropout is well approximated by the trajectory either
    side of it. A multi-hour dropout is not — interpolating across it would draw
    a straight line through a peak. Those fall back to the median for that
    segment at that time-of-day and day-of-week, which is the honest prior.
    """
    max_gap = cfg["cleaning"]["max_interpolate_gap"]
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    df["tod"] = df["timestamp"].dt.hour * 2 + df["timestamp"].dt.minute // 30
    df["dow"] = df["timestamp"].dt.dayofweek

    before_null = {c: int(df[c].isna().sum()) for c in SENSOR_COLS}

    for col in SENSOR_COLS:
        df[col] = (df.groupby("road_id", group_keys=False)[col]
                     .apply(lambda s: s.interpolate(method="linear",
                                                    limit=max_gap,
                                                    limit_area="inside")))

    interp_null = {c: int(df[c].isna().sum()) for c in SENSOR_COLS}

    for col in SENSOR_COLS:
        seasonal = df.groupby(["road_id", "dow", "tod"])[col].transform("median")
        segment = df.groupby("road_id")[col].transform("median")
        df[col] = df[col].fillna(seasonal).fillna(segment).fillna(df[col].median())

    # Fields that are plans or counts rather than measurements.
    df["signal_timing"] = (df["signal_timing"]
                           .fillna(df.groupby(["road_id", "tod"])["signal_timing"]
                                     .transform("median"))
                           .fillna(df.groupby("road_id")["signal_timing"]
                                     .transform("median")))
    # An absent window carries no *reported* incident. Filling with a rate would
    # fabricate accidents; 0 is the correct and conservative assumption.
    df["accident_count"] = df["accident_count"].fillna(0)

    stats["nulls_before_impute"] = before_null
    stats["nulls_interpolated"] = {c: before_null[c] - interp_null[c] for c in SENSOR_COLS}
    stats["nulls_seasonal_median"] = interp_null
    log.info("impute: interpolated %s, seasonal-median filled %s",
             stats["nulls_interpolated"], interp_null)
    return df


# --------------------------------------------------------------------------- #
# 6. Weather harmonisation
# --------------------------------------------------------------------------- #

def clean_weather(weather: pd.DataFrame, cfg: dict, stats: dict,
                  log) -> pd.DataFrame:
    """Normalise the label vocabulary and fill the small numeric gaps.

    The raw file carries 13 spellings of 4 conditions. 'Overcast' is not in the
    data dictionary's vocabulary but appears 358 times; it is mapped to Cloudy,
    which is the nearest documented concept. Unmapped labels are logged rather
    than silently passed through.
    """
    vocab = cfg["cleaning"]["weather_vocabulary"]
    raw_labels = weather["weather_condition"].nunique()

    key = weather["weather_condition"].astype(str).str.strip().str.lower()
    weather["weather_condition"] = key.map(vocab)

    unmapped = key[weather["weather_condition"].isna()].unique().tolist()
    if unmapped:
        log.warning("weather: %d unmapped label(s) -> 'Unknown': %s",
                    len(unmapped), unmapped)
        weather["weather_condition"] = weather["weather_condition"].fillna("Unknown")

    for col in ("temperature", "visibility"):
        n_missing = int(weather[col].isna().sum())
        weather[col] = (weather.groupby("station_id", group_keys=False)[col]
                               .apply(lambda s: s.interpolate(limit_direction="both")))
        stats.setdefault("weather_nulls_filled", {})[col] = n_missing
    weather["rainfall"] = weather["rainfall"].fillna(0.0)

    stats["weather_labels_before"] = int(raw_labels)
    stats["weather_labels_after"] = int(weather["weather_condition"].nunique())
    log.info("weather: collapsed %d raw labels to %d canonical conditions",
             raw_labels, weather["weather_condition"].nunique())
    return weather


def broadcast_weather(weather: pd.DataFrame, log) -> pd.DataFrame:
    """Expand hourly observations to both 30-minute windows of the hour."""
    weather = weather.copy()
    weather["hour_ts"] = weather["timestamp"].dt.floor("h")
    half = weather.copy()
    half["hour_ts"] = half["hour_ts"] + pd.Timedelta(minutes=30)
    out = pd.concat([weather, half], ignore_index=True)
    out = out.rename(columns={"hour_ts": "join_ts"}).drop(columns=["timestamp"])
    out = out.drop_duplicates(subset=["station_id", "join_ts"], keep="first")
    log.info("weather: broadcast to %d station x 30-min rows", len(out))
    return out


# --------------------------------------------------------------------------- #
# 7-9. Merge, derive, parse
# --------------------------------------------------------------------------- #

def merge_sources(traffic: pd.DataFrame, weather: pd.DataFrame,
                  calendar: pd.DataFrame, stats: dict, log) -> pd.DataFrame:
    """Left-join weather and calendar onto the traffic grain."""
    before = len(traffic)
    df = traffic.merge(
        weather, how="left",
        left_on=["weather_station_id", "timestamp"],
        right_on=["station_id", "join_ts"],
    ).drop(columns=["station_id", "join_ts"], errors="ignore")

    df["date"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    df = df.merge(cal, on="date", how="left")

    for flag in ("public_holiday", "event_flag", "roadwork_flag"):
        df[flag] = df[flag].fillna(0).astype("int8")
    df["holiday_name"] = df["holiday_name"].fillna("")
    df["event_name"] = df["event_name"].fillna("")

    assert len(df) == before, "merge changed the row count — check for key fan-out"
    stats["weather_match_rate"] = float(df["weather_condition"].notna().mean())
    stats["merged_rows"] = int(len(df))
    log.info("merge: %d rows, weather matched on %.2f%% of rows",
             len(df), 100 * stats["weather_match_rate"])
    return df


def derive_congestion(df: pd.DataFrame, cfg: dict, stats: dict,
                      log) -> pd.DataFrame:
    """Fill blank congestion_level from the volume-to-capacity ratio.

    Note for the modelling stage: this rule reproduces the *populated* labels
    with ~99.9% agreement, which means congestion_level is very nearly a
    deterministic function of traffic_volume. It is therefore a legitimate
    target but never a feature at the same timestamp.
    """
    capacity_30min = df["road_capacity"] / 2.0
    vc = df["traffic_volume"] / capacity_30min
    df["vc_ratio"] = vc

    bands = cfg["cleaning"]["congestion_bands"]
    edges = [b[0] for b in bands] + [bands[-1][1]]
    labels = [b[2] for b in bands]
    derived = pd.cut(vc, bins=edges, labels=labels, right=False,
                     include_lowest=True).astype(object)

    populated = df["congestion_level"].notna()
    agreement = float((derived[populated] == df.loc[populated, "congestion_level"]).mean())

    n_blank = int(df["congestion_level"].isna().sum())
    df["congestion_derived"] = df["congestion_level"].isna()
    df["congestion_level"] = df["congestion_level"].fillna(pd.Series(derived, index=df.index))

    stats["congestion_blank_filled"] = n_blank
    stats["congestion_rule_agreement"] = agreement
    log.info("congestion: derived %d blank labels; rule agrees with existing "
             "labels on %.2f%% of populated rows", n_blank, 100 * agreement)
    return df


def parse_vehicle_mix(df: pd.DataFrame, stats: dict, log) -> pd.DataFrame:
    """Expand the vehicle_type_dist JSON string into four share columns."""
    classes = ["2W", "Car", "LCV", "HCV"]

    def _parse(value):
        if not isinstance(value, str):
            return {}
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}

    parsed = df["vehicle_type_dist"].map(_parse)
    n_failed = int(sum(1 for p in parsed if not p) - df["vehicle_type_dist"].isna().sum())
    for cls in classes:
        col = f"share_{cls.lower()}"
        df[col] = parsed.map(lambda d: d.get(cls, np.nan)).astype("float32")
        df[col] = df[col].fillna(df[col].median())

    stats["vehicle_mix_parse_failures"] = max(n_failed, 0)
    log.info("vehicle mix: parsed into %d share columns", len(classes))
    return df.drop(columns=["vehicle_type_dist"])


def run(cfg: dict | None = None) -> dict:
    """Execute M2 end to end and persist the merged, cleaned table."""
    cfg = cfg or load_config()
    log = get_logger("clean", cfg)
    log.info("=== M2 Cleaning & Merge ===")
    stats: dict = {}

    interim = Path(cfg["paths"]["interim"])
    traffic = read_table(interim / "traffic_valid.parquet")
    weather = read_table(interim / "weather_valid.parquet")
    calendar = read_table(interim / "calendar_valid.parquet")

    traffic = deduplicate(traffic, stats, log)
    master = build_segment_master(traffic, log)
    traffic = rebuild_grid(traffic, master, cfg, stats, log)
    traffic = flag_statistical_outliers(traffic, cfg, stats, log)
    traffic = impute_sensors(traffic, cfg, stats, log)

    weather = clean_weather(weather, cfg, stats, log)
    weather = broadcast_weather(weather, log)

    df = merge_sources(traffic, weather, calendar, stats, log)
    df = derive_congestion(df, cfg, stats, log)
    df = parse_vehicle_mix(df, stats, log)

    # vehicle_count duplicates traffic_volume in 97.4% of rows — a perfect leak
    # into the primary regression target. Dropped here, not at model time, so it
    # cannot be reintroduced by accident downstream.
    df = df.drop(columns=["vehicle_count", "date", "time"], errors="ignore")
    stats["leak_columns_dropped"] = ["vehicle_count"]

    # travel_time is segment_length / avg_speed to within rounding. Kept as a
    # target, recorded here so the report states the dependency openly.
    speed_ok = df["avg_speed"] > 0
    df["segment_length_km"] = np.where(
        speed_ok, df["travel_time"] * df["avg_speed"] / 60.0, np.nan)
    seg_len = df.groupby("road_id")["segment_length_km"].transform("median")
    df["segment_length_km"] = seg_len
    df["travel_time"] = df["travel_time"].fillna(
        df["segment_length_km"] / df["avg_speed"].replace(0, np.nan) * 60.0)

    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    save_parquet(df, Path(cfg["paths"]["interim"]) / "merged_clean.parquet")
    save_json(stats, Path(cfg["paths"]["reports"]) / "clean_stats.json")

    log.info("M2 complete — merged table is %d rows x %d columns",
             len(df), df.shape[1])
    return stats


if __name__ == "__main__":
    run()
