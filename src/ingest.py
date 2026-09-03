"""M1 — Ingestion & Validation.

Loads the three raw tables, enforces schema and physical range checks, and
quarantines every rejected row rather than dropping it silently (NFR: 0 silent
drops). Load order follows the data dictionary: calendar, then weather, then
traffic.

Outputs
-------
data/interim/*_valid.parquet     rows that passed validation
data/interim/quarantine.parquet  rows that failed, with the reason
reports/ingest_stats.json        counters consumed by the data-quality report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_logger, load_config, save_json, save_parquet

EXPECTED_SCHEMA = {
    "traffic": [
        "road_id", "road_name", "latitude", "longitude", "weather_station_id",
        "date", "time", "traffic_volume", "vehicle_count", "vehicle_type_dist",
        "avg_speed", "occupancy", "congestion_level", "travel_time",
        "accident_count", "signal_timing", "road_capacity",
    ],
    "weather": [
        "station_id", "date", "time", "weather_condition",
        "temperature", "rainfall", "visibility",
    ],
    "calendar": [
        "date", "public_holiday", "holiday_name",
        "event_flag", "event_name", "roadwork_flag",
    ],
}


class SchemaError(RuntimeError):
    """Raised when a source file does not carry the columns the pipeline needs."""


def _check_schema(df: pd.DataFrame, table: str, log) -> None:
    expected = EXPECTED_SCHEMA[table]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        raise SchemaError(f"{table}: missing required columns {missing}")
    if extra:
        log.warning("%s: ignoring %d unexpected column(s): %s", table, len(extra), extra)


def _parse_datetime(dates: pd.Series, times: pd.Series, dayfirst: bool,
                    tz: str) -> pd.Series:
    """Combine a date and a time column into one tz-aware timestamp.

    The two source files disagree on date format — traffic is YYYY-MM-DD and
    weather is DD/MM/YYYY — so the caller passes dayfirst explicitly rather than
    letting pandas guess per row.
    """
    parsed_date = pd.to_datetime(dates, dayfirst=dayfirst, errors="coerce")
    combined = parsed_date.astype("datetime64[ns]") + pd.to_timedelta(
        times.astype(str).str.strip() + ":00", errors="coerce"
    )
    return combined.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")


def _range_flags(df: pd.DataFrame, ranges: dict) -> pd.Series:
    """Return a per-row reason string for rows violating a hard physical bound."""
    reasons = pd.Series("", index=df.index, dtype=object)
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        if bad.any():
            reasons.loc[bad] = reasons.loc[bad] + f"{col}_out_of_range;"
    return reasons


def load_calendar(cfg: dict, log) -> pd.DataFrame:
    path = Path(cfg["paths"]["raw"]) / cfg["data"]["calendar_file"]
    df = pd.read_csv(path)
    _check_schema(df, "calendar", log)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    for flag in ("public_holiday", "event_flag", "roadwork_flag"):
        df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype("int8")
    df["holiday_name"] = df["holiday_name"].fillna("").astype(str)
    df["event_name"] = df["event_name"].fillna("").astype(str)
    log.info("calendar: loaded %d dates (%d holidays, %d event days, %d roadwork days)",
             len(df), df.public_holiday.sum(), df.event_flag.sum(), df.roadwork_flag.sum())
    return df


def load_weather(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(cfg["paths"]["raw"]) / cfg["data"]["weather_file"]
    df = pd.read_csv(path)
    _check_schema(df, "weather", log)

    # Weather dates are DD/MM/YYYY — the opposite convention to the traffic file.
    df["timestamp"] = _parse_datetime(df["date"], df["time"], dayfirst=True,
                                      tz=cfg["project"]["timezone"])
    for col in ("temperature", "rainfall", "visibility"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    reasons = _range_flags(df, cfg["validation"]["ranges"])
    reasons.loc[df["timestamp"].isna()] += "unparseable_timestamp;"

    bad = reasons != ""
    quarantine = df.loc[bad].copy()
    quarantine["_reject_reason"] = reasons.loc[bad]
    quarantine["_source_table"] = "weather"

    valid = df.loc[~bad].drop(columns=["date", "time"])
    log.info("weather: %d rows loaded, %d quarantined, %d valid",
             len(df), len(quarantine), len(valid))
    return valid, quarantine


def load_traffic(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(cfg["paths"]["raw"]) / cfg["data"]["traffic_file"]
    df = pd.read_csv(path)
    _check_schema(df, "traffic", log)

    df["timestamp"] = _parse_datetime(df["date"], df["time"], dayfirst=False,
                                      tz=cfg["project"]["timezone"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    numeric = ["latitude", "longitude", "traffic_volume", "vehicle_count",
               "avg_speed", "occupancy", "travel_time", "accident_count",
               "signal_timing", "road_capacity"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["road_id"] = df["road_id"].astype(str).str.strip()
    df["congestion_level"] = df["congestion_level"].astype(str).str.strip()
    df.loc[df["congestion_level"].isin(["nan", "", "None"]), "congestion_level"] = np.nan

    reasons = _range_flags(df, cfg["validation"]["ranges"])
    reasons.loc[df["timestamp"].isna()] += "unparseable_timestamp;"
    reasons.loc[df["road_id"].isna() | (df["road_id"] == "")] += "missing_road_id;"

    bad = reasons != ""
    quarantine = df.loc[bad].copy()
    quarantine["_reject_reason"] = reasons.loc[bad]
    quarantine["_source_table"] = "traffic"

    valid = df.loc[~bad].copy()
    log.info("traffic: %d rows loaded, %d quarantined (out-of-range sensor faults), "
             "%d valid", len(df), len(quarantine), len(valid))
    log.info("traffic: %d segments, %s to %s", valid.road_id.nunique(),
             valid.timestamp.min(), valid.timestamp.max())
    return valid, quarantine


def run(cfg: dict | None = None) -> dict:
    """Execute M1 end to end and persist the interim tables."""
    cfg = cfg or load_config()
    log = get_logger("ingest", cfg)
    log.info("=== M1 Ingestion & Validation ===")

    interim = Path(cfg["paths"]["interim"])

    calendar = load_calendar(cfg, log)
    weather, weather_q = load_weather(cfg, log)
    traffic, traffic_q = load_traffic(cfg, log)

    quarantine = pd.concat([traffic_q, weather_q], ignore_index=True)

    save_parquet(calendar, interim / "calendar_valid.parquet")
    save_parquet(weather, interim / "weather_valid.parquet")
    save_parquet(traffic, interim / "traffic_valid.parquet")
    if len(quarantine):
        # Quarantined rows keep every original column so a reviewer can audit them.
        keep = [c for c in quarantine.columns if not c.startswith("_")] + \
               ["_reject_reason", "_source_table"]
        save_parquet(quarantine[keep].astype(str), interim / "quarantine.parquet")

    stats = {
        "traffic_rows_raw": int(len(traffic) + len(traffic_q)),
        "traffic_rows_valid": int(len(traffic)),
        "traffic_quarantined": int(len(traffic_q)),
        "weather_rows_raw": int(len(weather) + len(weather_q)),
        "weather_rows_valid": int(len(weather)),
        "weather_quarantined": int(len(weather_q)),
        "calendar_rows": int(len(calendar)),
        "quarantine_reasons": quarantine["_reject_reason"].value_counts().to_dict()
        if len(quarantine) else {},
        "segments": int(traffic.road_id.nunique()),
        "date_min": str(traffic.timestamp.min()),
        "date_max": str(traffic.timestamp.max()),
    }
    save_json(stats, Path(cfg["paths"]["reports"]) / "ingest_stats.json")
    log.info("M1 complete — quarantine holds %d rows", len(quarantine))
    return stats


if __name__ == "__main__":
    run()
