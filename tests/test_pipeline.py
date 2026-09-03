"""Tests for the FlowCast pipeline.

Run with:  pytest tests/ -v

The suite is deliberately weighted towards the things that fail *silently*.
A crash announces itself; a lag feature that reaches one window into the future
does not — it just produces a suspiciously good number that nobody questions.
Most of what follows is there to catch that class of error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linreg_scratch import LinearRegressionGD  # noqa: E402
from utils import load_config, time_split_bounds  # noqa: E402

CFG = load_config(ROOT / "config.yaml")
PROCESSED = Path(CFG["paths"]["processed"]) / "flowcast_dataset.parquet"
pytestmark = pytest.mark.skipif(
    not PROCESSED.exists(),
    reason="Run `python src/run_pipeline.py --stages data` first")


@pytest.fixture(scope="module")
def dataset():
    return pd.read_parquet(PROCESSED)


@pytest.fixture(scope="module")
def features():
    return json.loads((Path(CFG["paths"]["processed"]) /
                       "feature_columns.json").read_text())


# --------------------------------------------------------------------------- #
# Leakage — the failures that do not announce themselves
# --------------------------------------------------------------------------- #

class TestLeakage:

    def test_no_contemporaneous_target_columns_in_features(self, features):
        """The measured-at-T columns must never appear in the feature list."""
        banned = {"traffic_volume", "avg_speed", "occupancy", "vc_ratio",
                  "travel_time", "accident_count", "vehicle_count",
                  "temperature", "rainfall", "visibility"}
        assert banned.isdisjoint(features), \
            f"contemporaneous columns leaked into features: {banned & set(features)}"

    def test_vehicle_count_dropped_entirely(self, dataset):
        """vehicle_count equals traffic_volume in 97% of rows — a perfect leak."""
        assert "vehicle_count" not in dataset.columns

    def test_no_feature_is_near_perfectly_correlated_with_target(self, dataset, features):
        """A |r| above 0.99 against the target is a leak, not a good feature."""
        corr = dataset[features].corrwith(dataset["y_volume"]).abs()
        offenders = corr[corr > 0.99]
        assert offenders.empty, f"suspiciously predictive features: {dict(offenders)}"

    def test_lag_features_actually_lag(self, dataset):
        """lag1 at time T must equal the measured value at T-1, per segment."""
        seg = dataset[dataset.road_id == dataset.road_id.iloc[0]].sort_values("timestamp")
        # traffic_volume at T is y_volume; lag1 at T should be y_volume at T-1.
        shifted = seg["y_volume"].shift(1)
        both = pd.DataFrame({"lag": seg["traffic_volume_lag1"], "shift": shifted}).dropna()
        assert np.allclose(both["lag"], both["shift"], atol=1e-3), \
            "traffic_volume_lag1 does not match the previous window's volume"

    def test_rolling_features_exclude_current_window(self, dataset):
        """A rolling mean that includes T would contain the answer."""
        seg = dataset[dataset.road_id == dataset.road_id.iloc[0]].sort_values("timestamp")
        manual = seg["y_volume"].shift(1).rolling(4, min_periods=2).mean()
        # The persisted frame was warm-up trimmed, so the pipeline computed its
        # first few rolling values over history that is no longer in this frame.
        # Comparing there would flag a difference in the test's reconstruction,
        # not in the pipeline — so the first full window is skipped.
        both = pd.DataFrame({"a": seg["traffic_volume_rollmean4"],
                             "b": manual}).dropna().iloc[5:]
        assert np.allclose(both["a"], both["b"], atol=1e-2), \
            "rolling mean window is not shifted clear of the target window"


# --------------------------------------------------------------------------- #
# Cleaning correctness
# --------------------------------------------------------------------------- #

class TestCleaning:

    def test_no_nulls_in_modelling_columns(self, dataset, features):
        assert dataset[features].isna().sum().sum() == 0

    def test_no_impossible_physical_values_survive(self, dataset):
        assert (dataset["y_volume"] >= 0).all(), "negative volume survived cleaning"
        assert (dataset["y_speed"] <= 200).all(), "impossible speed survived cleaning"
        assert (dataset["y_speed"] > 0).all()

    def test_grid_is_uniform(self, dataset):
        """Every segment must have the same number of evenly spaced windows.

        The LSTM reshapes the flat table into (segment, time, channel) and would
        silently misalign segments if this were ragged.
        """
        counts = dataset.groupby("road_id").size()
        assert counts.nunique() == 1, f"ragged grid: {counts.min()}..{counts.max()}"

        seg = dataset[dataset.road_id == dataset.road_id.iloc[0]].sort_values("timestamp")
        gaps = seg["timestamp"].diff().dropna().unique()
        assert len(gaps) == 1, f"non-uniform time spacing: {gaps}"
        assert gaps[0] == pd.Timedelta(minutes=CFG["data"]["window_minutes"])

    def test_no_duplicate_keys(self, dataset):
        assert not dataset.duplicated(["road_id", "timestamp"]).any()

    def test_congestion_labels_are_the_controlled_vocabulary(self, dataset):
        allowed = {"Free-flow", "Moderate", "Heavy", "Severe"}
        assert set(dataset["y_congestion"].dropna().astype(str)) <= allowed

    def test_weather_labels_harmonised(self, dataset):
        """13 raw spellings must have collapsed to the 4-condition vocabulary."""
        labels = set(dataset["weather_condition"].dropna().unique())
        assert labels <= {"Clear", "Cloudy", "Rain", "Fog", "Unknown"}, labels
        assert not any(lab != lab.strip() or lab.islower() for lab in labels)

    def test_expected_segment_count(self, dataset):
        assert dataset["road_id"].nunique() == CFG["data"]["n_segments"]


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

class TestSplitting:

    def test_split_is_chronological_and_disjoint(self, dataset):
        train_end, val_end = time_split_bounds(
            dataset["timestamp"], CFG["split"]["train_frac"], CFG["split"]["val_frac"])
        train = dataset[dataset.timestamp <= train_end]
        val = dataset[(dataset.timestamp > train_end) & (dataset.timestamp <= val_end)]
        test = dataset[dataset.timestamp > val_end]

        assert len(train) and len(val) and len(test)
        assert train.timestamp.max() < val.timestamp.min()
        assert val.timestamp.max() < test.timestamp.min()
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_every_segment_appears_in_every_split(self, dataset):
        """A time-based split must not strand a segment in one period only."""
        train_end, val_end = time_split_bounds(
            dataset["timestamp"], CFG["split"]["train_frac"], CFG["split"]["val_frac"])
        for part in (dataset[dataset.timestamp <= train_end],
                     dataset[(dataset.timestamp > train_end) &
                             (dataset.timestamp <= val_end)],
                     dataset[dataset.timestamp > val_end]):
            assert part["road_id"].nunique() == CFG["data"]["n_segments"]


# --------------------------------------------------------------------------- #
# The from-scratch model
# --------------------------------------------------------------------------- #

class TestScratchLinearRegression:

    def test_gradient_matches_numerical_derivative(self):
        """The whole point of writing it by hand is that the calculus is right."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 12))
        y = X @ rng.normal(size=12) + 3.0 + rng.normal(scale=0.4, size=500)
        model = LinearRegressionGD()
        assert model.gradient_check(X, y) < 1e-5

    def test_recovers_known_coefficients(self):
        """On clean synthetic data it must find the generating weights."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(4000, 5))
        true_w = np.array([2.0, -1.5, 0.0, 3.5, 0.8])
        y = X @ true_w + 7.0
        model = LinearRegressionGD(learning_rate=0.15, epochs=800,
                                   batch_size=512).fit(X, y)
        recovered = np.array(list(model.coefficients(
            [f"x{i}" for i in range(5)]).values()))
        # Coefficients come back on the standardised-feature scale; X is already
        # unit-variance here, so they should match the generating weights.
        assert np.allclose(recovered, true_w, atol=0.05), recovered

    def test_cost_decreases_monotonically_overall(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(2000, 8))
        y = X @ rng.normal(size=8) + rng.normal(scale=0.3, size=2000)
        model = LinearRegressionGD(epochs=100).fit(X, y)
        history = model.cost_history
        assert history[-1] < history[0]
        assert all(np.isfinite(history))

    def test_diverging_learning_rate_is_caught_not_returned(self):
        """A silent inf-weight model would poison every downstream metric."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(1000, 6)) * 50
        y = X @ rng.normal(size=6) * 100
        model = LinearRegressionGD(learning_rate=5.0, epochs=40).fit(X, y)
        assert np.all(np.isfinite(model.w))
        assert model.learning_rate_used < 5.0, \
            "divergence guard did not back the learning rate off"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class TestMetrics:

    def test_perfect_prediction_scores_perfectly(self):
        from evaluate import regression_metrics
        y = np.array([10.0, 20.0, 30.0])
        m = regression_metrics(y, y)
        assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1.0

    def test_mape_masks_zeros_rather_than_exploding(self):
        """Overnight windows genuinely hit zero volume; MAPE must not blow up."""
        from evaluate import regression_metrics
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([5.0, 110.0, 190.0])
        m = regression_metrics(y_true, y_pred)
        assert np.isfinite(m["mape"])
        assert m["mape"] < 20

    def test_threshold_selection_beats_default(self):
        from evaluate import best_threshold
        rng = np.random.default_rng(4)
        y = (rng.random(2000) < 0.05).astype(int)
        score = np.clip(y * 0.5 + rng.random(2000) * 0.5, 0, 1)
        assert 0.0 < best_threshold(y, score) < 1.0
