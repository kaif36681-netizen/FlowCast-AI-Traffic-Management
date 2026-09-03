"""Shared utilities for the FlowCast pipeline.

Holds configuration loading, logging setup, deterministic seeding, and the small
IO helpers every module needs. Nothing here knows about traffic — keep domain
logic in the module that owns it.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve every path entry to an absolute path."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in cfg["paths"].items()}
    for directory in cfg["paths"].values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and to logs/flowcast.log.

    Every cleaning stage logs a line here; the data-quality report is assembled
    from the structured counters, not from parsing these logs, but the log is
    the audit trail a reviewer reads first.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
                            datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_dir = Path(cfg["paths"]["logs"]) if cfg else PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "flowcast.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches so a rerun reproduces the metrics."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:  # pyarrow absent — fall back to compressed CSV
        alt = path.with_suffix(".csv.gz")
        df.to_csv(alt, index=False, compression="gzip")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read parquet if present, else the .csv.gz fallback written by save_parquet."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    alt = path.with_suffix(".csv.gz")
    if alt.exists():
        # The CSV fallback loses dtypes; restore the datetime columns the
        # pipeline joins on, or every downstream merge fails on str vs datetime.
        df = pd.read_csv(alt)
        for col in ("timestamp", "date", "join_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
        return df
    raise FileNotFoundError(f"No table at {path} or {alt}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


# --------------------------------------------------------------------------- #
# Time-based splitting
# --------------------------------------------------------------------------- #


def time_split_bounds(timestamps: pd.Series, train_frac: float, val_frac: float
                      ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (train_end, val_end) cut points on the ordered timeline.

    Splitting is by calendar time, not by row index, so every segment gets the
    same train/val/test periods and no segment leaks across the boundary.
    """
    t_min, t_max = timestamps.min(), timestamps.max()
    span = t_max - t_min
    train_end = t_min + span * train_frac
    val_end = t_min + span * (train_frac + val_frac)
    return train_end, val_end


def split_frame(df: pd.DataFrame, ts_col: str, train_end, val_end):
    """Split a frame into train/val/test on the given cut points."""
    train = df[df[ts_col] <= train_end]
    val = df[(df[ts_col] > train_end) & (df[ts_col] <= val_end)]
    test = df[df[ts_col] > val_end]
    return train, val, test
