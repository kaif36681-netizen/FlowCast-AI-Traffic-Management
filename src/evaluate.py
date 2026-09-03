"""Evaluation metrics and the model scoreboard.

Every model — classical and deep — is scored through these functions on the
identical test window, so the head-to-head comparison in the final report is
like for like.

Metric choices follow the PRD:
    volume / travel time   RMSE headline, with MAE, MAPE and R2 alongside
    congestion (4-class)   macro-F1 headline, because the two classes that
                           matter operationally (Heavy, Severe) are the rare
                           ones and accuracy would be dominated by Free-flow
    accident risk (binary) ROC-AUC headline, with PR-AUC alongside — at a 0.9%
                           base rate ROC-AUC flatters a model, and average
                           precision is the metric that reflects what an
                           operator experiences when acting on the top-ranked
                           segments
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, MAPE and R2 for a continuous target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    # MAPE is undefined at zero. Traffic volume genuinely hits 0 on quiet
    # overnight windows, so the denominator is masked rather than epsilon-padded,
    # which would produce a meaningless spike instead of an honest exclusion.
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(resid[mask] / y_true[mask])) * 100) if mask.any() else float("nan")

    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    """Macro-F1 headline plus per-class precision/recall and the confusion matrix."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }
    if labels is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        out["confusion_matrix"] = cm.tolist()
        out["labels"] = list(labels)
        out["per_class"] = {
            lab: {
                "precision": float(precision_score(y_true, y_pred, labels=[i],
                                                   average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, labels=[i],
                                             average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, labels=[i],
                                     average="macro", zero_division=0)),
                "support": int((np.asarray(y_true) == i).sum()),
            }
            for i, lab in enumerate(labels)
        }
    return out


def binary_risk_metrics(y_true, y_score, threshold: float | None = None) -> dict:
    """ROC-AUC and PR-AUC, plus operating-point metrics at a chosen threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.sum() else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.sum() else float("nan"),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    if threshold is not None:
        y_hat = (y_score >= threshold).astype(int)
        out.update({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "flagged_rate": float(y_hat.mean()),
        })

    # Lift at the top decile: of the 10% of windows the model ranks riskiest,
    # how many times the base rate do they actually contain? This is the number
    # that tells a response coordinator whether the ranking is worth following.
    k = max(1, int(0.10 * len(y_score)))
    top_idx = np.argsort(-y_score)[:k]
    top_rate = float(y_true[top_idx].mean())
    out["top_decile_rate"] = top_rate
    out["top_decile_lift"] = float(top_rate / y_true.mean()) if y_true.mean() > 0 else float("nan")
    return out


def best_threshold(y_true, y_score, metric: str = "f1") -> float:
    """Pick the probability cut-off that maximises F1 on the validation split.

    Chosen on validation and then frozen — selecting it on test would be reading
    the answer sheet before the exam.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.50, 0.999, 120)))
    best, best_score = 0.5, -1.0
    for t in candidates:
        y_hat = (y_score >= t).astype(int)
        score = f1_score(y_true, y_hat, zero_division=0)
        if score > best_score:
            best, best_score = float(t), score
    return best


def error_breakdown(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                    by: list[str]) -> pd.DataFrame:
    """Break RMSE and MAE down by the given columns — the PRD's error analysis."""
    work = df.copy()
    work["_err"] = work[y_true_col] - work[y_pred_col]
    grouped = work.groupby(by, observed=True)["_err"]
    out = pd.DataFrame({
        "rmse": grouped.apply(lambda s: float(np.sqrt(np.mean(s ** 2)))),
        "mae": grouped.apply(lambda s: float(np.mean(np.abs(s)))),
        "bias": grouped.mean(),
        "n": grouped.size(),
    }).reset_index()
    return out


def build_scoreboard(results: dict) -> pd.DataFrame:
    """Flatten the nested results dict into one comparison table."""
    rows = []
    for target, models in results.items():
        for model_name, metrics in models.items():
            row = {"target": target, "model": model_name}
            row.update({k: v for k, v in metrics.items()
                        if isinstance(v, (int, float))})
            rows.append(row)
    return pd.DataFrame(rows)
