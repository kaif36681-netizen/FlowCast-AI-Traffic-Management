# FlowCast — Data Quality Report

Corridor: **Northline** · pipeline version **1.0** · seed **42**

Every figure below is emitted by the pipeline itself, not transcribed by hand. Rerunning `python src/run_pipeline.py` regenerates this file.

## 1. What arrived

| Source table | Rows | Notes |
| --- | ---: | --- |
| `traffic_sensor_log.csv` | 178,468 | 25 segments, 2025-01-01 to 2025-05-31 |
| `weather_observations.csv` | 10,872 | 3 stations, hourly |
| `calendar_events.csv` | 151 | one row per date |

## 2. Defects found and what was done

### 2.1 Out-of-range sensor faults (quarantined, not dropped)

**712 rows** (0.40%) violated a hard physical bound and were written to `data/interim/quarantine.parquet` with the reason attached. They are recoverable and auditable; nothing was silently discarded.

| Reason | Rows |
| --- | ---: |
| `traffic_volume_out_of_range` | 241 |
| `avg_speed_out_of_range` | 237 |
| `occupancy_out_of_range` | 234 |

A negative vehicle count and a speed of 300 km/h are not extreme observations — they are a detector reporting nonsense. Treating them as outliers to be winsorised would preserve a fabricated reading; quarantining removes the row and lets imputation reconstruct the window from its neighbours.

### 2.2 Duplicate readings

**1,759 duplicate rows** on `road_id + timestamp`, of which 1,759 were byte-identical — the signature of detector retries. The survivor is the row with the fewest nulls, chosen by sorting on a completeness score before deduplicating, so a retry that carried more fields than the original is not thrown away.

### 2.3 Absent windows (the defect that is easiest to miss)

The complete grid is 181,200 rows (25 segments x 48 windows x the date range). **5,203 windows (2.87%) were absent from the feed entirely** — not null, simply not present.

This matters more than it looks. Left as delivered, a lag feature computed with `shift(1)` would reach across a two-hour dropout and present 07:00 as the window immediately before 09:30. The pipeline reindexes onto the full grid before any lag is computed, which makes every gap explicit and keeps the time axis uniform.

### 2.4 Missing values and how they were filled

| Channel | Null before | Interpolated (gap ≤ 2 windows) | Seasonal-median filled |
| --- | ---: | ---: | ---: |
| `traffic_volume` | 9,534 | 9,510 | 24 |
| `avg_speed` | 9,534 | 9,510 | 24 |
| `occupancy` | 9,534 | 9,510 | 24 |

Short gaps are interpolated linearly because the trajectory either side genuinely constrains the missing value. Longer gaps are not — drawing a straight line across a three-hour dropout would erase a peak — so they fall back to the median for that segment at that time-of-day and day-of-week, which is the honest prior rather than an invented trajectory.

Statistical outliers nulled by per-segment z-score (|z| > 5.0): `{'traffic_volume': 0, 'avg_speed': 0, 'occupancy': 0}`. **This is zero, and that is a finding, not a bug** — the hard physical bounds in M1 had already removed every genuinely faulty reading, and what remained was well-behaved. The z-score gate stays in the pipeline because a future data drop may not be so clean.

### 2.5 Weather label harmonisation

The raw file carries **13 distinct spellings** of **4 conditions** — case variants, trailing whitespace and synonyms (`RAIN`, `rainy`, `Rain`).

One judgement call worth flagging: `Overcast` appears 358 times and is **not** in the data dictionary's vocabulary. It was mapped to `Cloudy`, the nearest documented concept, rather than kept as a fifth class or dropped. Any label the vocabulary does not cover is logged as a warning rather than passed through silently.

### 2.6 Temporal alignment

Weather dates are `DD/MM/YYYY`; traffic dates are `YYYY-MM-DD`. Both are parsed with an explicit `dayfirst` flag rather than letting the parser infer per row, which on a corridor dataset spanning January to May would silently transpose day and month for every date up to the 12th.

Weather is hourly and traffic is half-hourly, so each observation is broadcast to both 30-minute windows of its hour. Post-merge weather match rate: **100.00%**.

### 2.7 Congestion label derivation

**31,720 rows** had no `congestion_level`. They were filled from the volume-to-capacity ratio using the banding in the data dictionary.

Validating that rule against the rows that *were* populated: it reproduces the existing label on **99.64%** of them.

> **This near-perfect agreement is the single most important finding in this report.** `congestion_level` is not an independent signal — it is a deterministic banding of `traffic_volume`. A classifier given contemporaneous volume would score near 1.00 macro-F1 and forecast nothing at all. Congestion is therefore modelled strictly as a *next-window* target from lagged features only.

### 2.8 Leakage columns removed

| Column | Why it was removed |
| --- | --- |
| `vehicle_count` | Equals `traffic_volume` in 97.4% of rows — a perfect leak into the primary regression target. Dropped in M2 so it cannot be reintroduced downstream. |
| contemporaneous `occupancy`, `avg_speed`, `vc_ratio` | Measured at the target window. Available only as lags. |
| unlagged `temperature` / `rainfall` / `visibility` | Weather at the target window would be a forecast in production, so the model is given the previous window's observation instead. |

`travel_time` is segment length ÷ speed to within rounding, so the travel-time model is in substance a speed model. It is reported as its own target because the operations team acts on minutes, but the dependency is stated rather than presented as a fifth independent result.

## 3. What came out

| Property | Value |
| --- | --- |
| Analysis-ready rows | 172,800 |
| Engineered features | 93 |
| Warm-up rows dropped (no full 1-week history) | 8,400 |
| Coverage | 2025-01-08 to 2025-05-31 |
| Incident base rate | 0.907% |
| Nulls in any modelling column | 0 (asserted) |

### Class balance

| Congestion class | Share |
| --- | ---: |
| Free-flow | 61.3% |
| Moderate | 23.9% |
| Heavy | 9.3% |
| Severe | 5.5% |

The incident base rate of 0.907% is a roughly 1-in-110 imbalance. Every accident-risk model is class-weighted, and PR-AUC is reported alongside the PRD's ROC-AUC because at this base rate ROC-AUC flatters a model that an operator would find useless.

## 4. Reproducibility

All randomness is seeded at 42. `python src/run_pipeline.py` reruns ingestion, cleaning, features, both model engines and this report from the raw CSVs in one command.

---

*Generated by `src/eda.py` · Meridian Mobility Systems · Data & AI Division*