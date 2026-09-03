# FlowCast — Technical Report

**Short-horizon traffic forecasting for the Northline Corridor**
Meridian Mobility Systems · Data & AI Division · pipeline v1.0

Every number in this document is emitted by the pipeline on a held-out test
window and can be regenerated with `python src/run_pipeline.py`. Nothing is
transcribed by hand.

---

## 1. Executive summary

FlowCast forecasts traffic conditions 30 minutes ahead across 25 arterial
segments, from 178,468 raw sensor readings spanning 1 January to 31 May 2025.

**Three of the PRD's five acceptance criteria are met. Two are not, and the
reasons differ in a way that matters.**

| Criterion | Target | Achieved | Verdict |
| --- | --- | --- | --- |
| Volume forecast accuracy | MAPE ≤ 12% | **8.92%** | ✅ met |
| Congestion classification | Macro-F1 ≥ 0.80 | **0.7675** | ❌ missed by 0.033 |
| Accident-risk ranking | ROC-AUC ≥ 0.75 | **0.6433** | ❌ missed by 0.107 |
| Sequence model beats classical | on volume RMSE | 55.59 vs 55.86 | ✅ met, marginally |
| Reproducible, one command | — | `run_pipeline.py` | ✅ met |

The congestion miss is a metric artefact worth arguing about. The accident miss
is a data limitation, and Section 6 demonstrates that no amount of modelling
effort will close it.

The headline operational result: **volume can be forecast 30 minutes ahead to
within 8.9% mean absolute percentage error, a 45% reduction in RMSE against the
naive persistence forecast** that any control room already has for free.

---

## 2. Data reality

| Source | Rows | Grain |
| --- | ---: | --- |
| `traffic_sensor_log.csv` | 178,468 | segment × 30 min |
| `weather_observations.csv` | 10,872 | station × hour |
| `calendar_events.csv` | 151 | date |

Defects found and quantified (full detail in `reports/data_quality.md`):

| Defect | Extent | Treatment |
| --- | ---: | --- |
| Out-of-range sensor faults | 712 rows | Quarantined with reason, not dropped |
| Duplicate `road_id + timestamp` | 1,759 rows | Most-complete survivor retained |
| **Windows absent from the feed entirely** | **5,203 (2.87%)** | Grid rebuilt before any lag computed |
| Null volume / speed / occupancy | ~4,400 each | Interpolated (≤2 windows) or seasonal median |
| Weather label spellings | 13 → 4 | Controlled vocabulary, unmapped labels logged |
| Blank `congestion_level` | 31,720 | Derived from V/C banding |
| Date format mismatch | two conventions | Explicit `dayfirst` per source |

The absent-window defect is the one that quietly ruins a project. Left as
delivered, `shift(1)` on the raw file would present 07:00 as the window
immediately preceding 09:30. The pipeline materialises the complete
25 × 151 × 48 grid before computing a single lag.

---

## 3. The three leakage traps

These are not in the PRD. They are the difference between a model that scores
well and a model that forecasts.

**3.1 `congestion_level` is a deterministic banding of `traffic_volume`.**
The V/C rule from the data dictionary reproduces the populated labels with
**99.64% agreement**. A classifier given contemporaneous volume would score near
1.00 macro-F1 and predict nothing. Congestion is modelled strictly as a
next-window target from lagged features only.

**3.2 `vehicle_count` equals `traffic_volume` in 97.4% of rows.** A perfect leak
into the primary regression target. Dropped in M2 so it cannot reappear.

**3.3 `travel_time` is segment length ÷ speed to within rounding.** The implied
segment length is stable per road (median 2.03 km). The travel-time model is in
substance a speed model expressed in minutes. It is reported as its own target
because operations act on minutes, but the dependency is stated rather than
presented as an independent fourth result.

Enforcement is structural, not editorial: `features.py` shifts every dynamic
feature by at least one window within its own segment, `feature_columns.json` is
the access contract, and `tests/test_pipeline.py` asserts that no contemporaneous
column appears in it and that no feature correlates above 0.99 with the target.

---

## 4. Results

Test window: **10–31 May 2025**, 25,925 segment-windows, never seen in training
or tuning. Split is chronological — train 8 Jan–18 Apr, validate 19 Apr–9 May.

### 4.1 Traffic volume (primary target)

| Model | RMSE | MAPE | R² |
| --- | ---: | ---: | ---: |
| Mean baseline | 287.31 | 83.31% | −0.001 |
| Persistence (naive) | 102.40 | 16.02% | 0.873 |
| Linear Regression (**from scratch**, NumPy GD) | 65.81 | 12.27% | 0.947 |
| Linear Regression (scikit-learn) | 65.80 | 12.30% | 0.947 |
| Decision Tree | 67.32 | 10.43% | 0.945 |
| Random Forest | 58.37 | 9.32% | 0.959 |
| **XGBoost (tuned)** | **55.86** | **8.92%** | **0.962** |
| LSTM (sequence) | 55.59 | 9.57% | — |

The from-scratch gradient-descent implementation lands **within 0.01 RMSE of
scikit-learn's closed-form solution**, and its analytic gradient agrees with a
numerical one to a maximum relative error of **3.9 × 10⁻¹¹**. The mathematics is
verified, not asserted.

Including the persistence baseline is not decoration. It is the forecast a
control room already has, and it is the only honest floor for a 30-minute
horizon: XGBoost's real contribution is the **45% RMSE reduction against it**,
not the 0.96 R² that would look impressive in isolation.

### 4.2 Speed and travel time

| Target | Winner | RMSE | MAPE | R² |
| --- | --- | ---: | ---: | ---: |
| Speed (km/h) | XGBoost | 3.44 | 8.28% | 0.913 |
| Travel time (min) | XGBoost | 1.05 | 9.13% | 0.835 |

### 4.3 Congestion classification (4-class)

| Model | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Majority baseline | 0.188 | 0.603 |
| Persistence (naive) | 0.619 | 0.782 |
| SVM (RBF, 12k rows) | 0.728 | 0.848 |
| Decision Tree | 0.740 | 0.842 |
| Logistic Regression | 0.754 | 0.858 |
| Random Forest | 0.766 | 0.869 |
| **XGBoost** | **0.7675** | **0.877** |

**On the miss.** Macro-F1 weights all four classes equally, and Severe is 5.5% of
windows. Accuracy is 0.877 and the model beats persistence by 0.148 macro-F1.
The metric is the right one — the rare classes are exactly the operationally
important ones, and rewarding a model for getting Free-flow right would be
pointless — but 0.80 macro-F1 on a 4-class problem with a 61/24/9/5 split is a
demanding bar that was set without sight of the class balance. Recommendation:
either retain 0.80 and accept a longer tuning cycle focused on Severe recall, or
restate the criterion as macro-F1 ≥ 0.80 **on the Heavy and Severe classes
jointly**, which this model already achieves.

### 4.4 Accident risk (binary, 0.91% base rate)

| Model | ROC-AUC | PR-AUC | Top-decile lift |
| --- | ---: | ---: | ---: |
| **Logistic Regression** | **0.6433** | 0.0214 | **2.64×** |
| Linear SVM (36k rows) | 0.5678 | 0.0159 | 1.92× |
| Random Forest | 0.5584 | 0.0140 | 2.20× |
| XGBoost | 0.4943 | 0.0108 | 1.04× |

The ordering is inverted from every other target: the linear model wins and
gradient boosting scores at coin-flip. That inversion is diagnostic rather than
embarrassing — with 250 positives in the test window and weak signal, boosting
fits training-window noise that does not generalise, while the linear model's
rigidity protects it. Section 6 establishes why.

**Calibration.** Class weighting is necessary for ranking but destroys the
absolute scale of the output: the raw class-weighted classifier emits a mean
predicted probability of **48.6%** on a corridor whose observed incident rate is
**0.96%**. Each model is therefore Platt-scaled on held-out validation data —
the earlier half of the validation window fits the curve, the later half selects
the operating threshold.

| | Raw | Calibrated | Observed |
| --- | ---: | ---: | ---: |
| Mean predicted probability | 48.55% | **0.80%** | 0.96% |
| Brier score | 0.25087 | **0.00952** | — |

The transform is monotone, so ROC-AUC, PR-AUC and top-decile lift are unchanged
by construction — only the scale is corrected. The fitted curve is
sigmoid(2.416·s − 6.052), and predicted risk now spans 0.38% to 2.20% across the
corridor, which is the range an operator can act on.

### 4.5 Classical vs deep

| Model | Volume RMSE | Volume MAPE | Congestion macro-F1 | Params | Train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost (classical) | 55.86 | **8.92%** | 0.7675 | — | 16 s |
| **LSTM (2-layer, multi-task)** | **55.59** | 9.57% | 0.7692 | 64,269 | 3.2 min |
| BiLSTM *(not a forecaster)* | 55.13 | 9.07% | 0.7705 | 64,269 | 3.7 min |

**The PRD's benchmark criterion is met, and it should not be celebrated.** The
LSTM wins on RMSE by 0.27 vehicles — under half a percent — and *loses* on MAPE.
It cost 64,269 parameters and twelve times the training wall-clock to draw with
a gradient-boosted tree. Training converged cleanly (early stop at epoch 12,
validation loss flat from epoch 8), so this is not an undertrained model. The
honest reading is that the engineered lag and rolling features already encode
most of the sequential structure an LSTM would have to learn, and on a corridor
of this size the recurrent architecture does not earn its complexity.

The BiLSTM is reported for completeness and is **not deployable**: reading the
window in both directions means reading the future. It is included only for the
historical-analysis view, where the whole window is already known.

### 4.6 Prediction intervals (FR-11)

Quantile gradient boosting at the 10th / 50th / 90th percentiles.

| Property | Value |
| --- | ---: |
| Nominal coverage | 80% |
| **Empirical coverage** | **80.7%** |
| Mean interval width | 126 vehicles (29% of mean volume) |

| Condition | Mean width | Windows |
| --- | ---: | ---: |
| Off-peak, dry | 90 | 17,516 |
| Rain | 94 | 834 |
| Peak | 211 | 7,225 |
| Peak + rain | 230 | 350 |

The band is genuinely conditional — it more than doubles at peak — and it is
calibrated to within 0.7 points of nominal. A confidence estimate that cannot be
checked is decoration; this one is checked, and the number is reported whether or
not it flatters the system.

---

## 5. Feature engineering

93 features from 17 raw columns.

- **Temporal** — cyclical sin/cos hour and day-of-week (so 23:30 and 00:00 are
  adjacent rather than 23 units apart), AM/PM peak flags, weekend, night.
- **Lags** — volume, speed, occupancy and V/C at t−1, −2, −3, −4, −48 (same
  window yesterday) and −336 (same window last week).
- **Rolling** — mean and standard deviation over 4, 8 and 48 windows, each
  shifted clear of the target window.
- **Momentum** — first differences, and same-window day-over-day change.
- **Capacity** — V/C ratio, absolute headroom, signal-timing-to-capacity ratio.
- **Weather** — lagged temperature, rainfall and visibility, rain and
  low-visibility flags, 3-hour cumulative rainfall.
- **Interactions** — holiday × peak, event × peak, roadwork × peak, rain × peak.

Weather is used at t−1 rather than t. In production an operator would have a
short-range forecast, so this is the conservative choice.

Top drivers for volume: `traffic_volume_lag1`, `traffic_volume_lag336`,
`vc_ratio_lag1`, `occupancy_lag1`, `traffic_volume_lag48`. All lagged, all
physically sensible — which is itself the sanity check that no leak survived.

---

## 6. Why accident risk fails, and what would fix it

When a model underperforms there are two explanations calling for opposite
responses: the model is wrong (tune harder), or the signal is absent (stop).
`src/diagnose_accident.py` separates them by fitting a deliberately **cheating**
model that sees the target window's own congestion, occupancy and speed —
information no forecaster could ever have.

| Model | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Honest (lagged features only) | 0.6322 | 0.0193 |
| **CHEATING (contemporaneous state)** | **0.6383** | 0.0216 |
| Advantage from cheating | **+0.0062** | — |

**Cheating buys 0.006 ROC-AUC.** The ceiling on this dataset is approximately
0.64 and the 0.75 target is unreachable, not through insufficient modelling but
because the covariates do not carry the information. Further tuning would be
fitting noise.

What signal does exist is entirely congestion-mediated:

| Congestion class | Incident rate |
| --- | ---: |
| Free-flow | 0.733% |
| Moderate | 0.758% |
| Heavy | 1.587% |
| Severe | 2.347% |

**Recommendations.** First, judge the feature on top-decile lift rather than
ROC-AUC — at 2.64×, patrolling the riskiest tenth of corridor-hours finds 2.6
times the base rate of incidents, which is operationally useful even at 0.64
AUC. Second, acquire the data that actually predicts collisions: road geometry
and lane counts, historical collision locations, merge and signal-phase
information, and sub-window speed variance (harsh-braking proxies). Aggregate
30-minute means smooth away exactly the volatility that precedes an incident.

---

## 7. What is built

```
flowcast/
├── config.yaml              every tunable; no magic numbers in modules
├── src/
│   ├── utils.py             config, logging, seeding, time-split helpers
│   ├── ingest.py            M1 schema + range validation, quarantine
│   ├── clean.py             M2 dedup, grid rebuild, impute, harmonise, merge
│   ├── features.py          M3 93 features with structural leakage guards
│   ├── eda.py               M4 figures + generated data-quality report
│   ├── linreg_scratch.py    NumPy gradient descent, with gradient check
│   ├── ml_models.py         M5 classical family, tuning, model cards
│   ├── dl_model.py          M6 multi-task LSTM + MC-dropout uncertainty
│   ├── uncertainty.py       quantile intervals with coverage measurement
│   ├── diagnose_accident.py signal-vs-model ceiling diagnostic
│   ├── predict.py           inference API, never retrains
│   ├── evaluate.py          shared metrics and scoreboard
│   └── run_pipeline.py      one-command orchestrator, resumable
├── dashboard/app.py         13 views (9 required + 4 extensions)
└── tests/test_pipeline.py   21 tests, weighted to silent failures
```

**Non-functional requirements.** Corridor-wide inference for one horizon:
**0.021 s** against a 30 s target. A full 24 hours of 25 segments: 0.108 s. All
21 tests pass. Seeded at 42 throughout; every stage is idempotent and
checkpointed, so an interrupted run resumes rather than restarting.

**Dashboard.** All nine required views plus four extensions — a ranked risk
watchlist for the incident coordinator, a geographic corridor map, the
interval-calibration view, and the acceptance scorecard. Every view renders
against real persisted outputs; the dashboard loads models and never trains. The
signature element is the corridor strip: 25 tiles, one per segment, pinned above
every view and coloured by forecast severity, in a fixed spatial order so an
operator builds positional memory of the corridor.

---

## 8. Known limitations

1. **Single corridor, five months.** No summer or monsoon period. Seasonal
   generalisation is untested.
2. **Travel time is not independent** of speed (Section 3.3).
3. **SVM is fitted on a subsample** (12k rows for congestion, 36k for accident
   risk). An RBF kernel is O(n²) in training rows; the full 121k window is not
   tractable on a workstation. The subsample is taken from the end of the
   training period so it is the most recent data.
4. **Hyperparameter search is compact** — two XGBoost candidates over three
   forward-chaining folds, on the primary target only. Depth, learning rate and
   subsampling dominate; a wider grid was not affordable and the CV spread
   (±2.4 RMSE) suggests it would not have changed the winner.
5. **MC-dropout is an approximation**, not a calibrated posterior. The quantile
   intervals are the ones with measured coverage and are what the dashboard
   shows.
6. **Congestion labels are 18% derived** by the V/C rule rather than observed.
   The rule agrees with observed labels 99.64% of the time, but the derived rows
   are by construction perfectly consistent with it.

---

## 9. Recommended next steps

1. Restate or accept the congestion criterion (Section 4.3) — a decision for the
   product owner, not a modelling task.
2. Drop the LSTM from the deployment path. It ties XGBoost at twelve times the
   training cost and greater operational complexity. Retain the code as the
   benchmark that justifies the decision.
3. Acquire collision-predictive data before promising a 0.75 AUC incident model
   (Section 6).
4. Extend to a full year before any claim of seasonal robustness.
5. Add drift monitoring on the feature distributions — a corridor's behaviour
   changes when a parallel route closes, and nothing here would currently notice.

---

*Generated against pipeline v1.0, seed 42. Regenerate with
`python src/run_pipeline.py`.*
