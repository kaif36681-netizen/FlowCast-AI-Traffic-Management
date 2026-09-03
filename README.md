# FlowCast

Short-horizon traffic forecasting for the Northline Corridor — 25 arterial
segments, 30-minute windows, four prediction targets, and an operations
dashboard.

Given the last few hours of sensor readings, weather and the events calendar,
FlowCast forecasts the next 30 minutes: traffic volume, mean speed, travel time,
congestion class and the probability of a reported incident — each with a
confidence estimate that has been checked against outcomes rather than asserted.

**Headline result: volume forecast to 8.92% MAPE, a 45% RMSE reduction against
the naive persistence forecast.** Full results, including two acceptance
criteria that were *not* met and why, are in
[`reports/technical_report.md`](reports/technical_report.md).

---

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline: raw CSVs -> cleaned data -> features -> models -> reports
python src/run_pipeline.py

# Dashboard (reads persisted artefacts; never trains)
streamlit run dashboard/app.py
```

Expect roughly 25–35 minutes end to end on a single core, dominated by the
Random Forest fits and the LSTM. Every stage is checkpointed, so an interrupted
run resumes rather than restarting.

Place the three source files in `data/raw/` first:

```
data/raw/traffic_sensor_log.csv
data/raw/weather_observations.csv
data/raw/calendar_events.csv
```

---

## Running parts of it

```bash
python src/run_pipeline.py --stages data      # M1–M4: ingest, clean, features, EDA
python src/run_pipeline.py --stages models    # M5, M6, prediction intervals
python src/run_pipeline.py --skip deep        # everything except the LSTM
python src/run_pipeline.py --only clean features
python src/run_pipeline.py --input new_export.csv   # score a fresh sensor export

python src/ml_models.py --targets y_volume    # retrain one target
python src/dl_model.py --variants lstm        # one sequence variant
python src/diagnose_accident.py               # signal-vs-model ceiling check

pytest tests/ -v                              # 21 tests
```

---

## How it fits together

```
data/raw/*.csv
     │
     ├─ M1  ingest.py      schema + physical range checks; failures quarantined
     │                     with a reason, never silently dropped
     ├─ M2  clean.py       dedup → rebuild the complete time grid → outliers →
     │                     impute → harmonise weather → merge → derive labels
     ├─ M3  features.py    93 features, every dynamic one lagged ≥ 1 window
     ├─ M4  eda.py         figures + generated reports/data_quality.md
     ├─ M5  ml_models.py   linear (from scratch + sklearn), tree, forest,
     │                     XGBoost, SVM — with naive baselines for the floor
     ├─ M6  dl_model.py    multi-task LSTM, MC-dropout uncertainty
     ├─      uncertainty.py quantile intervals + measured coverage
     └─ M7  dashboard/     13 views over persisted outputs
```

Configuration lives entirely in `config.yaml`. No magic numbers in the modules.

---

## The part worth reading first

Three properties of this dataset will inflate your metrics if you miss them, and
none are mentioned in the source documentation:

1. **`congestion_level` is a deterministic banding of `traffic_volume`** — the
   V/C rule reproduces the observed labels with 99.64% agreement. Feed a
   classifier contemporaneous volume and it will score near-perfectly while
   forecasting nothing.
2. **`vehicle_count` equals `traffic_volume`** in 97.4% of rows.
3. **`travel_time` is segment length ÷ speed** to within rounding.

And one defect that is easy to miss: **5,203 windows (2.87%) are absent from the
feed entirely** — not null, simply not present. Computing `shift(1)` on the raw
file would present 07:00 as the window immediately before 09:30.

All four are handled structurally rather than by convention — see
`features.py` for the shift discipline and `tests/test_pipeline.py::TestLeakage`
for the assertions that keep it that way.

---

## Reports

| File | What it contains |
| --- | --- |
| `reports/technical_report.md` | Results, acceptance scorecard, limitations |
| `reports/data_quality.md` | Every cleaning decision, quantified (generated) |
| `reports/classical_results.json` | Full metrics for every classical model |
| `reports/deep_results.json` | LSTM/BiLSTM metrics and training curves |
| `reports/accident_diagnostic.json` | Signal-vs-model ceiling analysis |
| `reports/uncertainty.json` | Interval coverage and width by condition |
| `models/model_cards.json` | Per-model provenance, windows, metrics |
| `reports/figures/` | EDA figures as PNG |

---

## Requirements

Python 3.11+. See `requirements.txt`. PyTorch is CPU-only by default; the LSTM
trains in about 3 minutes on one core.

---

## Layout

```
config.yaml              all tunables
src/                     pipeline modules (see diagram above)
dashboard/app.py         Streamlit dashboard, 13 views
tests/test_pipeline.py   21 tests, weighted towards silent failures
data/raw/                source CSVs (read-only)
data/interim/            quarantine, validated and merged tables
data/processed/          analysis-ready dataset and predictions
models/                  fitted models, scaler, model cards
reports/                 reports, metrics, figures
```
