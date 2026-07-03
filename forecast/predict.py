"""Stage 2 - load persisted forecasters and produce predictions.

Thin helper used by the EMS stages (baseline / generative / agentic). It loads
the best model per target saved by `forecast.train`, rebuilds the exact feature
columns those models expect, and returns predictions aligned to the input rows.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

import config
from forecast.train import TARGETS, _make_supervised
from utils.logger import get_logger

log = get_logger("forecast.predict")


@lru_cache(maxsize=None)
def load_forecaster(target: str) -> dict:
    """Load and cache the persisted best model bundle for one target."""
    path = config.MODELS_DIR / f"{target}_best.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py forecast` first."
        )
    return joblib.load(path)


def predict_frame(df: pd.DataFrame, target: str) -> pd.Series:
    """Return predictions for `target` aligned to `df`'s rows (NaN where lags
    are undefined, i.e. the first week of the series)."""
    bundle = load_forecaster(target)
    supervised, features = _make_supervised(df.reset_index(drop=True), target)
    # _make_supervised drops rows with undefined lags; map predictions back by
    # position onto the tail of the original frame.
    preds = bundle["model"].predict(supervised[features])
    out = pd.Series(index=df.index, dtype=float)
    out.iloc[len(df) - len(preds):] = preds
    return out


def add_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a `<target>_forecast` column per target."""
    result = df.copy()
    for target in TARGETS:
        result[f"{target}_forecast"] = predict_frame(df, target)
    return result
