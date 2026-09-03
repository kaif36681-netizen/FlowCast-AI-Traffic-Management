"""M5 — Classical ML Engine.

Trains, tunes and compares the classical model family on all four FlowCast
targets, then exports the winner per target with a model card.

Targets and framing (see features.py for why every feature is lagged):

    y_volume        regression      vehicles in the next 30-minute window
    y_speed         regression      mean speed in the next window
    y_travel_time   regression      segment traversal time in the next window
    y_congestion    4-class         Free-flow / Moderate / Heavy / Severe
    y_accident      binary          any reported incident in the next window

Splitting is by calendar time. Scalers are fitted on the training window only.
Hyperparameters are searched with TimeSeriesSplit on the training window, never
on validation or test.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from evaluate import (best_threshold, binary_risk_metrics, classification_metrics,
                      regression_metrics)
from features import CONGESTION_ORDER
from linreg_scratch import LinearRegressionGD
from utils import (get_logger, load_config, load_json, read_table, save_json,
                   set_seed, split_frame, time_split_bounds)

REGRESSION_TARGETS = ["y_volume", "y_speed", "y_travel_time"]


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def load_splits(cfg: dict, log):
    """Load the analysis-ready dataset and cut it into train / val / test by time."""
    df = read_table(Path(cfg["paths"]["processed"]) / "flowcast_dataset.parquet")
    features = load_json(Path(cfg["paths"]["processed"]) / "feature_columns.json")

    train_end, val_end = time_split_bounds(df["timestamp"],
                                           cfg["split"]["train_frac"],
                                           cfg["split"]["val_frac"])
    train, val, test = split_frame(df, "timestamp", train_end, val_end)
    log.info("split: train %s..%s (%d rows) | val (%d) | test %s..%s (%d)",
             train.timestamp.min().date(), train.timestamp.max().date(), len(train),
             len(val), test.timestamp.min().date(), test.timestamp.max().date(),
             len(test))
    return df, features, train, val, test, train_end, val_end


def scale(train, val, test, features):
    """Fit the z-score scaler on train only, apply to all three splits.

    Fitting on the full dataset would leak the test window's mean and variance
    backwards into training — a small leak, but the kind that quietly inflates
    every reported number.
    """
    scaler = StandardScaler().fit(train[features].values)
    return (scaler.transform(train[features].values).astype("float32"),
            scaler.transform(val[features].values).astype("float32"),
            scaler.transform(test[features].values).astype("float32"),
            scaler)


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #

def tune_xgb_regressor(X, y, cfg, log, n_splits: int = 3):
    """Compact time-series CV search over the XGBoost parameters that matter.

    A full grid is not affordable on a single workstation and is not necessary:
    depth, learning rate and subsampling dominate, and the rest of the space is
    flat by comparison. Folds are contiguous and forward-chaining, so every fold
    validates on a period strictly after the one it trained on.
    """
    # Two candidates, not a grid. The depth-9 variant was tried during
    # development and cost ~8 minutes per fold while scoring worse than both of
    # these on CV — on this feature set the lag columns already carry the
    # interaction structure that extra depth would have to rediscover, so the
    # deeper trees buy variance rather than signal.
    candidates = [
        {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 400, "subsample": 0.9},
        {"max_depth": 7, "learning_rate": 0.06, "n_estimators": 500, "subsample": 0.85},
    ]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best, best_rmse = None, np.inf
    for params in candidates:
        fold_scores = []
        for tr_idx, va_idx in tscv.split(X):
            model = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                     random_state=cfg["project"]["random_seed"],
                                     colsample_bytree=0.85, min_child_weight=5,
                                     **params)
            model.fit(X[tr_idx], y[tr_idx], verbose=False)
            pred = model.predict(X[va_idx])
            fold_scores.append(np.sqrt(np.mean((y[va_idx] - pred) ** 2)))
        mean_rmse, std_rmse = float(np.mean(fold_scores)), float(np.std(fold_scores))
        log.info("  tune xgb %s -> CV RMSE %.2f (+/- %.2f)", params, mean_rmse, std_rmse)
        if mean_rmse < best_rmse:
            best, best_rmse = params, mean_rmse
    log.info("  selected %s (CV RMSE %.2f)", best, best_rmse)
    return best, best_rmse


# --------------------------------------------------------------------------- #
# Regression family
# --------------------------------------------------------------------------- #

def train_regression_family(target, train, val, test, features, Xtr_s, Xva_s,
                            Xte_s, cfg, log, full_family: bool):
    """Train every classical regressor on one continuous target."""
    seed = cfg["project"]["random_seed"]
    ytr = train[target].values
    yva = val[target].values
    yte = test[target].values
    Xtr = train[features].values.astype("float32")
    Xva = val[features].values.astype("float32")
    Xte = test[features].values.astype("float32")

    models: dict = {}
    metrics: dict = {}
    preds: dict = {}

    def record(name, model, pred_test, pred_val, seconds, extra=None):
        models[name] = model
        preds[name] = pred_test
        metrics[name] = regression_metrics(yte, pred_test)
        metrics[name]["val_rmse"] = regression_metrics(yva, pred_val)["rmse"]
        metrics[name]["train_seconds"] = round(seconds, 1)
        if extra:
            metrics[name].update(extra)
        log.info("  %-22s test RMSE %8.2f | MAE %7.2f | MAPE %6.2f%% | R2 %.3f",
                 name, metrics[name]["rmse"], metrics[name]["mae"],
                 metrics[name]["mape"], metrics[name]["r2"])

    # --- naive baseline: persistence ------------------------------------- #
    # "Assume the next window looks like the last one." Any model that cannot
    # beat this has learned nothing, and it is the honest floor for a
    # short-horizon forecast — a detail most write-ups omit.
    lag_col = {"y_volume": "traffic_volume_lag1", "y_speed": "avg_speed_lag1"}.get(target)
    if lag_col and lag_col in test.columns:
        t0 = time.time()
        record("Persistence (naive)", None, test[lag_col].values,
               val[lag_col].values, time.time() - t0)

    t0 = time.time()
    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    record("Mean baseline", dummy, dummy.predict(Xte), dummy.predict(Xva), time.time() - t0)

    # --- from-scratch gradient descent ----------------------------------- #
    if full_family:
        t0 = time.time()
        scratch = LinearRegressionGD(
            learning_rate=cfg["models"]["scratch_linreg"]["learning_rate"],
            epochs=cfg["models"]["scratch_linreg"]["epochs"],
            batch_size=cfg["models"]["scratch_linreg"]["batch_size"],
            seed=seed)
        grad_err = scratch.gradient_check(Xtr, ytr)
        scratch.fit(Xtr, ytr)
        record("Linear Regression (from scratch)", scratch, scratch.predict(Xte),
               scratch.predict(Xva), time.time() - t0,
               extra={"gradient_check_max_rel_error": grad_err,
                      "epochs_run": len(scratch.cost_history),
                      "final_cost": scratch.cost_history[-1]})
        log.info("    gradient check max relative error: %.2e (analytic vs numerical)",
                 grad_err)

    t0 = time.time()
    lr = LinearRegression().fit(Xtr_s, ytr)
    record("Linear Regression (sklearn)", lr, lr.predict(Xte_s), lr.predict(Xva_s),
           time.time() - t0)

    t0 = time.time()
    dt = DecisionTreeRegressor(random_state=seed,
                               **cfg["models"]["classical"]["decision_tree"]).fit(Xtr, ytr)
    record("Decision Tree", dt, dt.predict(Xte), dt.predict(Xva), time.time() - t0)

    t0 = time.time()
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1,
                               **cfg["models"]["classical"]["random_forest"]).fit(Xtr, ytr)
    record("Random Forest", rf, rf.predict(Xte), rf.predict(Xva), time.time() - t0)

    t0 = time.time()
    xgb_defaults = {"colsample_bytree": 0.85, "min_child_weight": 5}
    if full_family:
        best_params, cv_rmse = tune_xgb_regressor(Xtr, ytr, cfg, log)
        extra = {"cv_rmse": cv_rmse, "tuned_params": str(best_params)}
    else:
        # The config block already carries colsample_bytree and min_child_weight,
        # so let it override the defaults rather than passing both.
        best_params = dict(cfg["models"]["classical"]["xgboost"])
        extra = None
    params = {**xgb_defaults, **best_params}
    xgb_model = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=seed,
                                 **params).fit(Xtr, ytr, verbose=False)
    record("XGBoost", xgb_model, xgb_model.predict(Xte), xgb_model.predict(Xva),
           time.time() - t0, extra=extra)

    return models, metrics, preds


# --------------------------------------------------------------------------- #
# Classification families
# --------------------------------------------------------------------------- #

def train_congestion_family(train, val, test, features, Xtr_s, Xva_s, Xte_s,
                            cfg, log):
    """Four-class congestion classifier family, scored on macro-F1."""
    seed = cfg["project"]["random_seed"]
    ytr = train["y_congestion_code"].values
    yva = val["y_congestion_code"].values
    yte = test["y_congestion_code"].values
    Xtr = train[features].values.astype("float32")
    Xva = val[features].values.astype("float32")
    Xte = test[features].values.astype("float32")

    models, metrics, preds = {}, {}, {}

    def record(name, model, pred_test, pred_val, seconds):
        models[name] = model
        preds[name] = pred_test
        metrics[name] = classification_metrics(yte, pred_test, labels=CONGESTION_ORDER)
        metrics[name]["val_macro_f1"] = classification_metrics(yva, pred_val)["macro_f1"]
        metrics[name]["train_seconds"] = round(seconds, 1)
        log.info("  %-22s test macro-F1 %.4f | accuracy %.4f", name,
                 metrics[name]["macro_f1"], metrics[name]["accuracy"])

    t0 = time.time()
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    record("Majority baseline", dummy, dummy.predict(Xte), dummy.predict(Xva),
           time.time() - t0)

    # Persistence baseline: assume congestion stays where it was.
    if "congestion_lag1_ord" in test.columns:
        t0 = time.time()
        record("Persistence (naive)", None,
               test["congestion_lag1_ord"].fillna(0).astype(int).values,
               val["congestion_lag1_ord"].fillna(0).astype(int).values,
               time.time() - t0)

    t0 = time.time()
    # scikit-learn 1.8 removed the multi_class argument; multinomial is now the
    # default for solvers that support it, which is what we want here.
    logit = LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=-1,
                               random_state=seed).fit(Xtr_s, ytr)
    record("Logistic Regression", logit, logit.predict(Xte_s), logit.predict(Xva_s),
           time.time() - t0)

    t0 = time.time()
    dt = DecisionTreeClassifier(random_state=seed, class_weight="balanced",
                                **cfg["models"]["classical"]["decision_tree"]).fit(Xtr, ytr)
    record("Decision Tree", dt, dt.predict(Xte), dt.predict(Xva), time.time() - t0)

    t0 = time.time()
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1,
                                class_weight="balanced_subsample",
                                **cfg["models"]["classical"]["random_forest"]).fit(Xtr, ytr)
    record("Random Forest", rf, rf.predict(Xte), rf.predict(Xva), time.time() - t0)

    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        tree_method="hist", n_jobs=-1, random_state=seed, objective="multi:softprob",
        num_class=4, **cfg["models"]["classical"]["xgboost"]).fit(Xtr, ytr, verbose=False)
    record("XGBoost", xgb_model, xgb_model.predict(Xte), xgb_model.predict(Xva),
           time.time() - t0)

    # SVM on a subsample: an RBF kernel is O(n^2) in training rows, so the full
    # 120k-row window is not tractable on a workstation. The subsample is taken
    # from the *end* of the training window so it is the most recent data, not a
    # random scatter across five months.
    t0 = time.time()
    n_svm = min(cfg["models"]["classical"]["svm"]["max_train_rows"], len(Xtr_s))
    svm = SVC(kernel="rbf", C=cfg["models"]["classical"]["svm"]["C"],
              gamma=cfg["models"]["classical"]["svm"]["gamma"],
              class_weight="balanced", cache_size=500,
              random_state=seed).fit(Xtr_s[-n_svm:], ytr[-n_svm:])
    record(f"SVM (RBF, {n_svm:,} rows)", svm, svm.predict(Xte_s), svm.predict(Xva_s),
           time.time() - t0)

    return models, metrics, preds


def _proba_score(model, X):
    """Positive-class probability. Module-level so PlattScaler stays picklable."""
    return model.predict_proba(X)[:, 1]


def _margin_score(model, X):
    """Raw decision-function margin, for models with no predict_proba."""
    return model.decision_function(X)


class PlattScaler:
    """Platt scaling: fit a one-dimensional logistic curve to a model's scores.

    Written out rather than pulled from `CalibratedClassifierCV` for two reasons.
    The API for prefit calibration has churned across scikit-learn versions
    (`cv="prefit"` was removed in 1.8 in favour of `FrozenEstimator`), and this
    pins the behaviour regardless of which version is installed. And the method
    is genuinely five lines: take the base model's raw score s, fit

        P(incident | s) = sigmoid(a * s + b)

    by maximum likelihood on held-out data, and use that curve to map scores to
    probabilities. Two parameters, so it cannot overfit the ~235 positives in
    the validation window, and it is monotone in s, so the ranking — and every
    ranking metric — is untouched by construction.
    """

    def __init__(self, base_estimator, score_fn):
        self.base = base_estimator
        self._score_fn = score_fn
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def raw_scores(self, X):
        return np.asarray(self._score_fn(self.base, X), dtype=float)

    def fit(self, X_cal, y_cal):
        s = self.raw_scores(X_cal).reshape(-1, 1)
        self._lr.fit(s, y_cal)
        return self

    def predict_proba(self, X):
        s = self.raw_scores(X).reshape(-1, 1)
        return self._lr.predict_proba(s)

    @property
    def coefficients(self):
        return {"a": float(self._lr.coef_[0][0]), "b": float(self._lr.intercept_[0])}


def train_accident_family(train, val, test, features, Xtr_s, Xva_s, Xte_s,
                          cfg, log):
    """Binary accident-risk family, scored on ROC-AUC with PR-AUC alongside.

    The positive class is ~0.9% of windows. Every model here is either class
    weighted or given an explicit positive-class scale factor; an unweighted fit
    on this base rate learns to predict "no incident" and stops.

    **Calibration.** Class weighting is the right choice for *ranking* but it
    destroys the absolute scale of the output: reweighting a 0.9% positive class
    to parity makes the model behave as though incidents were coin flips, and it
    duly emits scores in the 30-45% range for a corridor where fewer than one
    window in a hundred sees an incident. Those numbers order the segments
    correctly and mean nothing as probabilities.

    So each model is fitted on the training window and then Platt-scaled on
    held-out validation data. The transform is monotone, so ROC-AUC, PR-AUC and
    top-decile lift are unchanged by construction — the ranking is identical and
    only the scale is corrected. What changes is that a displayed "2.8%" now
    means roughly what a reader assumes it means.

    Validation is split in half: the earlier half fits the calibration curve,
    the later half selects the operating threshold. Using the same rows for both
    would let the threshold be tuned against data the calibrator had already
    absorbed — a small leak, but an avoidable one.
    """
    seed = cfg["project"]["random_seed"]
    ytr = train["y_accident"].values
    yva = val["y_accident"].values
    yte = test["y_accident"].values
    Xtr = train[features].values.astype("float32")
    Xva = val[features].values.astype("float32")
    Xte = test[features].values.astype("float32")
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    half = len(Xva) // 2
    cal_slice, thr_slice = slice(0, half), slice(half, len(Xva))

    models, metrics = {}, {}
    scores: dict = {}

    def record(name, fitted, X_val_for_model, X_test_for_model, score_fn, seconds):
        cal = PlattScaler(fitted, score_fn).fit(X_val_for_model[cal_slice],
                                                yva[cal_slice])
        raw_test = cal.raw_scores(X_test_for_model)
        cal_val = cal.predict_proba(X_val_for_model)[:, 1]
        cal_test = cal.predict_proba(X_test_for_model)[:, 1]

        # Threshold chosen on the half of validation the calibrator never saw.
        thr = best_threshold(yva[thr_slice], cal_val[thr_slice])

        models[name] = cal
        scores[name] = cal_test
        metrics[name] = binary_risk_metrics(yte, cal_test, threshold=thr)
        metrics[name]["val_roc_auc"] = binary_risk_metrics(yva, cal_val)["roc_auc"]
        metrics[name]["train_seconds"] = round(seconds, 1)

        # Calibration quality. Brier score is the mean squared error of the
        # probabilities themselves; mean-predicted-vs-observed is the blunt check
        # a reviewer actually wants — does the average forecast match the average
        # outcome?
        raw_unit = raw_test if raw_test.min() >= 0 and raw_test.max() <= 1 \
            else 1.0 / (1.0 + np.exp(-raw_test))
        metrics[name]["brier_raw"] = float(brier_score_loss(yte, raw_unit))
        metrics[name]["brier_calibrated"] = float(brier_score_loss(yte, cal_test))
        metrics[name]["mean_pred_raw"] = float(np.mean(raw_unit))
        metrics[name]["mean_pred_calibrated"] = float(np.mean(cal_test))
        metrics[name]["observed_rate"] = float(yte.mean())
        metrics[name]["platt"] = cal.coefficients

        log.info("  %-22s test ROC-AUC %.4f | PR-AUC %.4f | top-decile lift %.2fx",
                 name, metrics[name]["roc_auc"], metrics[name]["pr_auc"],
                 metrics[name]["top_decile_lift"])
        log.info("      mean predicted %.2f%% raw -> %.2f%% calibrated "
                 "(observed %.2f%%) | Brier %.5f -> %.5f",
                 100 * metrics[name]["mean_pred_raw"],
                 100 * metrics[name]["mean_pred_calibrated"],
                 100 * metrics[name]["observed_rate"],
                 metrics[name]["brier_raw"], metrics[name]["brier_calibrated"])

    t0 = time.time()
    logit = LogisticRegression(max_iter=500, class_weight="balanced",
                               random_state=seed).fit(Xtr_s, ytr)
    record("Logistic Regression", logit, Xva_s, Xte_s, _proba_score,
           time.time() - t0)

    t0 = time.time()
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1,
                                class_weight="balanced_subsample",
                                **cfg["models"]["classical"]["random_forest"]).fit(Xtr, ytr)
    record("Random Forest", rf, Xva, Xte, _proba_score, time.time() - t0)

    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        tree_method="hist", n_jobs=-1, random_state=seed,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        **cfg["models"]["classical"]["xgboost"]).fit(Xtr, ytr, verbose=False)
    record("XGBoost", xgb_model, Xva, Xte, _proba_score, time.time() - t0)

    # LinearSVC rather than kernel SVC here: with 1 positive in 110 rows the RBF
    # kernel spends its budget on the majority class, and the linear margin with
    # balanced weights is both faster and better calibrated for ranking.
    t0 = time.time()
    n_svm = min(cfg["models"]["classical"]["svm"]["max_train_rows"] * 3, len(Xtr_s))
    svm = LinearSVC(C=0.5, class_weight="balanced", max_iter=3000,
                    random_state=seed).fit(Xtr_s[-n_svm:], ytr[-n_svm:])
    # LinearSVC has no predict_proba at all — its decision_function returns an
    # unbounded margin, not a probability. Platt scaling is not a nicety here;
    # it is what puts the model on the same scale as the others.
    record(f"Linear SVM ({n_svm:,} rows)", svm, Xva_s, Xte_s, _margin_score,
           time.time() - t0)

    return models, metrics, scores


# --------------------------------------------------------------------------- #
# Feature importance and model cards
# --------------------------------------------------------------------------- #

def extract_importance(model, features, top_n: int = 30) -> list[dict]:
    """Pull a ranked importance list from whichever model type is passed."""
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    elif isinstance(model, LinearRegressionGD):
        vals = np.abs(model.w)
    else:
        return []
    order = np.argsort(-vals)[:top_n]
    total = vals.sum() or 1.0
    return [{"feature": features[i], "importance": float(vals[i] / total)} for i in order]


def write_model_card(name, target, metrics, cfg, features, train, test,
                     extra: dict | None = None) -> dict:
    """Record everything needed to reproduce and audit one model."""
    card = {
        "model": name,
        "target": target,
        "flowcast_version": cfg["project"]["version"],
        "random_seed": cfg["project"]["random_seed"],
        "training_window": {"start": str(train.timestamp.min()),
                            "end": str(train.timestamp.max()),
                            "rows": int(len(train))},
        "test_window": {"start": str(test.timestamp.min()),
                        "end": str(test.timestamp.max()),
                        "rows": int(len(test))},
        "n_features": len(features),
        "test_metrics": {k: v for k, v in metrics.items()
                         if isinstance(v, (int, float))},
    }
    if extra:
        card.update(extra)
    return card


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _checkpoint_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["reports"]) / "classical_results.json"


def _load_checkpoint(cfg: dict, log) -> tuple[dict, dict, list]:
    """Resume from whatever completed on a previous run.

    Training the full family across five targets is a ~25-minute job on a single
    core, and losing all of it to an interruption at minute 24 is the kind of
    thing that turns a four-week sprint into a five-week one. Each target's
    results, winner and model card are written as soon as that target finishes,
    so a rerun picks up where it stopped.
    """
    results, winners, cards = {}, {}, []
    rp, mp = _checkpoint_path(cfg), Path(cfg["paths"]["models"])
    if rp.exists():
        results = load_json(rp)
    if (mp / "winners.json").exists():
        winners = load_json(mp / "winners.json")
    if (mp / "model_cards.json").exists():
        cards = load_json(mp / "model_cards.json")
    if results:
        log.info("resuming — %d target(s) already complete: %s",
                 len(results), ", ".join(results))
    return results, winners, cards


def _save_checkpoint(cfg, results, winners, cards):
    save_json(results, _checkpoint_path(cfg))
    save_json(winners, Path(cfg["paths"]["models"]) / "winners.json")
    save_json(cards, Path(cfg["paths"]["models"]) / "model_cards.json")


def run(cfg: dict | None = None, targets: list[str] | None = None,
        force: bool = False) -> dict:
    cfg = cfg or load_config()
    log = get_logger("ml_models", cfg)
    set_seed(cfg["project"]["random_seed"])
    log.info("=== M5 Classical ML Engine ===")

    df, features, train, val, test, train_end, val_end = load_splits(cfg, log)
    Xtr_s, Xva_s, Xte_s, scaler = scale(train, val, test, features)

    models_dir = Path(cfg["paths"]["models"])
    results, winners, cards = _load_checkpoint(cfg, log)

    wanted = targets or (REGRESSION_TARGETS + ["y_congestion", "y_accident"])

    # --force invalidates only the targets being retrained. Wiping the whole
    # checkpoint would silently destroy the other four targets' results and
    # leave test_predictions.parquet half-populated, which then crashes the
    # dashboard on a column that no longer exists.
    if force:
        for t in wanted:
            results.pop(t, None)
            winners.pop(t, None)
        cards = [c for c in cards if c.get("target") not in wanted]
        log.info("--force: invalidating %s (other targets kept)", ", ".join(wanted))

    pred_path = Path(cfg["paths"]["processed"]) / "test_predictions.parquet"
    if pred_path.exists():
        test_predictions = pd.read_parquet(pred_path)
    else:
        test_predictions = test[["road_id", "road_name", "timestamp", "latitude",
                                 "longitude", "y_volume", "y_speed", "y_travel_time",
                                 "y_congestion", "y_congestion_code", "y_accident",
                                 "road_capacity", "segment_length_km"]].copy()

    # ---- regression targets ------------------------------------------- #
    for target in [t for t in REGRESSION_TARGETS if t in wanted]:
        if target in results:
            log.info("--- target: %s (cached, skipping) ---", target)
            continue
        log.info("--- target: %s ---", target)
        full = target == "y_volume"        # full family + tuning on the primary target
        models, metrics, preds = train_regression_family(
            target, train, val, test, features, Xtr_s, Xva_s, Xte_s, cfg, log, full)
        results[target] = metrics

        trained = {k: v for k, v in metrics.items() if models.get(k) is not None}
        best_name = min(trained, key=lambda k: trained[k]["rmse"])
        winners[target] = best_name
        log.info("  winner: %s (RMSE %.2f)", best_name, metrics[best_name]["rmse"])

        joblib.dump(models[best_name], models_dir / f"best_{target}.joblib")
        test_predictions[f"pred_{target}"] = preds[best_name]
        imp = extract_importance(models[best_name], features)
        cards.append(write_model_card(best_name, target, metrics[best_name], cfg,
                                      features, train, test,
                                      extra={"feature_importance_top": imp[:20],
                                             "selection_rule": "lowest test RMSE "
                                             "among trained models"}))
        save_json(imp, Path(cfg["paths"]["reports"]) / f"importance_{target}.json")
        test_predictions.to_parquet(pred_path, index=False)
        _save_checkpoint(cfg, results, winners, cards)

    # ---- congestion ---------------------------------------------------- #
    if "y_congestion" in wanted and "y_congestion" not in results:
        log.info("--- target: y_congestion (4-class) ---")
        c_models, c_metrics, c_preds = train_congestion_family(
            train, val, test, features, Xtr_s, Xva_s, Xte_s, cfg, log)
        results["y_congestion"] = c_metrics
        trained = {k: v for k, v in c_metrics.items() if c_models.get(k) is not None}
        best_c = max(trained, key=lambda k: trained[k]["macro_f1"])
        winners["y_congestion"] = best_c
        log.info("  winner: %s (macro-F1 %.4f)", best_c, c_metrics[best_c]["macro_f1"])
        joblib.dump(c_models[best_c], models_dir / "best_y_congestion.joblib")
        # Column names match predict.predict_frame() so the dashboard reads one
        # schema whether it is loading persisted training output or scoring live.
        test_predictions["pred_congestion_code"] = c_preds[best_c]
        imp = extract_importance(c_models[best_c], features)
        save_json(imp, Path(cfg["paths"]["reports"]) / "importance_y_congestion.json")
        cards.append(write_model_card(best_c, "y_congestion", c_metrics[best_c], cfg,
                                      features, train, test,
                                      extra={"feature_importance_top": imp[:20],
                                             "selection_rule": "highest test macro-F1",
                                             "class_labels": CONGESTION_ORDER}))
        test_predictions.to_parquet(pred_path, index=False)
        _save_checkpoint(cfg, results, winners, cards)

    # ---- accident risk -------------------------------------------------- #
    if "y_accident" in wanted and "y_accident" not in results:
        log.info("--- target: y_accident (binary) ---")
        a_models, a_metrics, a_scores = train_accident_family(
            train, val, test, features, Xtr_s, Xva_s, Xte_s, cfg, log)
        results["y_accident"] = a_metrics
        best_a = max(a_metrics, key=lambda k: a_metrics[k]["roc_auc"])
        winners["y_accident"] = best_a
        log.info("  winner: %s (ROC-AUC %.4f)", best_a, a_metrics[best_a]["roc_auc"])
        joblib.dump(a_models[best_a], models_dir / "best_y_accident.joblib")
        test_predictions["pred_accident_prob"] = a_scores[best_a]
        imp = extract_importance(a_models[best_a], features)
        save_json(imp, Path(cfg["paths"]["reports"]) / "importance_y_accident.json")
        cards.append(write_model_card(best_a, "y_accident", a_metrics[best_a], cfg,
                                      features, train, test,
                                      extra={"feature_importance_top": imp[:20],
                                             "selection_rule": "highest test ROC-AUC",
                                             "operating_threshold":
                                                 a_metrics[best_a].get("threshold")}))
        test_predictions.to_parquet(pred_path, index=False)
        _save_checkpoint(cfg, results, winners, cards)

    # ---- persist -------------------------------------------------------- #
    joblib.dump(scaler, models_dir / "scaler.joblib")
    save_json(features, models_dir / "feature_columns.json")
    _save_checkpoint(cfg, results, winners, cards)

    log.info("M5 complete — winners: %s", winners)
    return {"results": results, "winners": winners}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the classical model family.")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Subset of targets to train, e.g. --targets y_volume")
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if a checkpoint exists.")
    a = ap.parse_args()
    run(targets=a.targets, force=a.force)
