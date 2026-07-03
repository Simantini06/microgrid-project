"""Stage 2 - train and select day-ahead forecasting models.

For each generation/demand target we train three regressors
(RandomForest, GradientBoosting, XGBoost), evaluate MAE and RMSE on a
time-ordered hold-out split, compare them against a seasonal-naive baseline,
and persist the best model per target for the later EMS stages to consume.

**Forecasting framing (important, and documented honestly).**
In Stage 1 the generation targets are deterministic functions of the weather
columns, so feeding same-hour weather in as features would let a model recover
them almost perfectly - an unrealistic result. Real day-ahead energy management
does *not* have perfect future weather; it forecasts from the calendar and from
recently observed load/generation. So the feature set here is:

* cyclical calendar encodings (hour, month, day-of-week) + weekend flag, and
* lagged actuals of the target itself: 24 h ago, 168 h (one week) ago, and a
  24 h rolling mean.

This yields an honest spread of difficulty: demand has a strong learnable
pattern, solar is moderate (cloud variability is irreducible here), and wind is
genuinely hard (near-random) - exactly the realistic picture the downstream
comparison needs. Targets, features, and metrics are never fabricated; they are
computed from the Stage 1 CSV and reproducible.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import config
from utils.logger import get_logger

log = get_logger("forecast")

DATASET_CSV = config.DATA_DIR / "microgrid_dataset.csv"
METRICS_JSON = config.MODELS_DIR / "forecast_metrics.json"
METRICS_MD = config.REPORTS_DIR / "forecast_metrics.md"

# Targets we forecast. Grid price is a *known* deterministic TOU tariff, not a
# stochastic quantity, so it is not forecast here.
TARGETS = ["solar_power_kw", "wind_power_kw", "demand_kw"]

# Fraction of the (time-ordered) series held out for testing.
_TEST_FRACTION = 0.20
_LAGS = (24, 168)          # one day, one week
_ROLL_WINDOW = 24


def _model_zoo() -> dict[str, object]:
    """Return a fresh set of candidate regressors (fixed seeds)."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            random_state=42, verbosity=0,
        ),
    }


def _make_supervised(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature matrix + target column for one forecast target.

    Adds cyclical calendar features and lagged/rolling features of the target,
    then drops the initial rows whose lags are undefined.
    """
    out = pd.DataFrame(index=df.index)

    # Cyclical calendar encodings (avoid false ordinal distance, e.g. 23->0h).
    out["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    out["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    out["is_weekend"] = df["is_weekend"].to_numpy()

    # Lagged actuals + rolling mean (shifted so no look-ahead leakage).
    for lag in _LAGS:
        out[f"lag_{lag}"] = df[target].shift(lag)
    out[f"roll_{_ROLL_WINDOW}"] = df[target].shift(1).rolling(_ROLL_WINDOW).mean()

    out[target] = df[target].to_numpy()
    out = out.dropna().reset_index(drop=True)

    features = [c for c in out.columns if c != target]
    return out, features


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(_rmse(y_true, y_pred), 4),
    }


def _train_target(df: pd.DataFrame, target: str) -> dict:
    """Train the model zoo for one target; return metrics + persist the best."""
    data, features = _make_supervised(df, target)
    split = int(len(data) * (1 - _TEST_FRACTION))
    train, test = data.iloc[:split], data.iloc[split:]
    x_train, y_train = train[features], train[target]
    x_test, y_test = test[features], test[target]

    # Seasonal-naive baseline: "same hour yesterday" (the lag_24 column).
    naive = _evaluate(y_test.to_numpy(), test["lag_24"].to_numpy())
    log.info("[%s] naive(24h) baseline: MAE=%.3f RMSE=%.3f",
             target, naive["mae"], naive["rmse"])

    scores: dict[str, dict[str, float]] = {}
    fitted: dict[str, object] = {}
    for name, model in _model_zoo().items():
        model.fit(x_train, y_train)
        metrics = _evaluate(y_test.to_numpy(), model.predict(x_test))
        scores[name] = metrics
        fitted[name] = model
        log.info("[%s] %-16s MAE=%.3f RMSE=%.3f",
                 target, name, metrics["mae"], metrics["rmse"])

    best_name = min(scores, key=lambda n: scores[n]["rmse"])
    best_path = config.MODELS_DIR / f"{target}_best.joblib"
    joblib.dump(
        {"model": fitted[best_name], "features": features,
         "target": target, "algorithm": best_name},
        best_path,
    )
    log.info("[%s] best = %s  ->  %s", target, best_name, best_path.name)

    return {
        "target": target,
        "n_train": len(train),
        "n_test": len(test),
        "features": features,
        "naive_baseline": naive,
        "models": scores,
        "best_model": best_name,
        "best_metrics": scores[best_name],
        "beats_naive_rmse": bool(scores[best_name]["rmse"] < naive["rmse"]),
    }


def _write_report(results: list[dict]) -> None:
    """Write a human-readable Markdown metrics report to reports/."""
    lines = [
        "# Forecast Model Report - Microgrid_AI (Stage 2)",
        "",
        "Day-ahead forecasts from calendar + lagged-actual features "
        "(no same-hour weather; see `forecast/train.py` for the rationale).",
        f"Time-ordered hold-out: last {int(_TEST_FRACTION * 100)} % of the series.",
        "",
        "| Target | Best model | MAE | RMSE | Naive RMSE | Beats naive? |",
        "| ------ | ---------- | --- | ---- | ---------- | ------------ |",
    ]
    for r in results:
        lines.append(
            f"| `{r['target']}` | {r['best_model']} | "
            f"{r['best_metrics']['mae']:.3f} | {r['best_metrics']['rmse']:.3f} | "
            f"{r['naive_baseline']['rmse']:.3f} | "
            f"{'yes' if r['beats_naive_rmse'] else 'no'} |"
        )
    lines += ["", "## All candidates (RMSE)", ""]
    for r in results:
        row = " / ".join(f"{n}: {m['rmse']:.3f}" for n, m in r["models"].items())
        lines.append(f"- **{r['target']}** - {row}")
    lines.append("")
    METRICS_MD.write_text("\n".join(lines), encoding="utf-8")


def train_all() -> list[dict]:
    """Train + select forecasters for every target; persist models + metrics."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} not found. Run `python main.py data` first."
        )

    df = pd.read_csv(DATASET_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded %d rows for forecasting.", len(df))

    results = [_train_target(df, t) for t in TARGETS]

    METRICS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_report(results)
    log.info("Wrote metrics -> %s and report -> %s", METRICS_JSON, METRICS_MD)

    log.info("Summary (best RMSE per target):")
    for r in results:
        log.info("  %-16s %-16s RMSE=%.3f (naive %.3f)",
                 r["target"], r["best_model"],
                 r["best_metrics"]["rmse"], r["naive_baseline"]["rmse"])
    return results
