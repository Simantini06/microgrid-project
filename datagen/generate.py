"""Stage 1 - build the synthetic microgrid operating dataset.

No suitable, redistributable real dataset couples *all* of the signals this
project needs (co-located solar, wind, demand, weather, and time-of-use price
for one site at hourly resolution). Rather than stitch mismatched sources or
fabricate "real" numbers, this module generates a transparent SYNTHETIC dataset
from documented physical models. Every assumption is stated here and echoed in
the generated data dictionary, and the whole file is reproducible from a fixed
seed (`--seed`).

Physical models used (all standard, defensible first-order approximations):

* Solar irradiance - clear-sky solar geometry (declination + hour angle for the
  site latitude) attenuated by hourly cloud cover.
* Solar power       - PV rating x (irradiance / 1000) with a mild high-temp
  derate and a fixed performance ratio.
* Wind power        - standard turbine power curve (cut-in / rated / cut-out)
  driven by a Weibull-distributed wind speed.
* Demand            - base load + morning/evening peaks + weekend and seasonal
  (summer cooling) factors + noise.
* Grid price        - Indian-style time-of-use tariff (off-peak / normal / peak)
  anchored to GRID_IMPORT_PRICE.

The battery is a *control* asset: its state of charge (SoC) is decided by the
EMS in later stages, so it is NOT a column here. Battery specifications live in
`config.py` (BATTERY_*).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from utils.logger import get_logger

log = get_logger("datagen")

# Output locations.
DATASET_CSV = config.DATA_DIR / "microgrid_dataset.csv"
DATA_DICTIONARY_MD = config.DATA_DIR / "data_dictionary.md"

# Generation tuning constants (documented, not tuned to any target result).
_START_DATE = "2024-01-01"            # arbitrary non-leap-adjacent start
_PV_PERFORMANCE_RATIO = 0.80          # soiling, wiring, inverter losses
_PV_TEMP_DERATE_PER_C = 0.004         # -0.4 %/degC above 25 degC (typical c-Si)
_CLEAR_SKY_PEAK_WM2 = 1000.0          # peak clear-sky irradiance
_DEMAND_BASE_KW = 30.0               # overnight base load
_DEMAND_NOISE_KW = 2.0               # gaussian measurement/behaviour noise


def _solar_elevation_sin(day_of_year: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """Return sin(solar elevation) via standard solar geometry (>=0 by day)."""
    lat = np.radians(config.SITE_LATITUDE_DEG)
    # Declination (deg) - Cooper's equation.
    decl = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + day_of_year))))
    # Hour angle (deg): 0 at solar noon, 15 deg per hour.
    hour_angle = np.radians(15.0 * (hour - 12.0))
    sin_elev = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    return np.clip(sin_elev, 0.0, None)


def _solar_power_kw(irradiance: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    """Convert plane-of-array irradiance + cell temp to PV output (kW)."""
    derate = 1.0 - _PV_TEMP_DERATE_PER_C * np.clip(temperature_c - 25.0, 0.0, None)
    power = (
        config.PV_RATED_KWP
        * (irradiance / _CLEAR_SKY_PEAK_WM2)
        * _PV_PERFORMANCE_RATIO
        * derate
    )
    return np.clip(power, 0.0, config.PV_RATED_KWP)


def _wind_power_kw(wind_speed: np.ndarray) -> np.ndarray:
    """Standard turbine power curve -> kW for the given wind speeds (m/s)."""
    cut_in, rated, cut_out = (
        config.WIND_CUT_IN_MS,
        config.WIND_RATED_MS,
        config.WIND_CUT_OUT_MS,
    )
    power = np.zeros_like(wind_speed)
    # Cubic ramp between cut-in and rated speed.
    ramp = (wind_speed >= cut_in) & (wind_speed < rated)
    power[ramp] = config.WIND_RATED_KW * (
        (wind_speed[ramp] ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)
    )
    # Flat at rated power between rated and cut-out speed.
    flat = (wind_speed >= rated) & (wind_speed <= cut_out)
    power[flat] = config.WIND_RATED_KW
    # Above cut-out: turbine parks (already zero).
    return power


def _time_of_use_price(hour: np.ndarray) -> np.ndarray:
    """Indian-style TOU tariff anchored to config.GRID_IMPORT_PRICE (peak)."""
    peak = config.GRID_IMPORT_PRICE
    price = np.full(hour.shape, 0.75 * peak)          # normal band default
    price[(hour >= 22) | (hour < 6)] = 0.50 * peak    # off-peak (night)
    price[((hour >= 9) & (hour < 12)) | ((hour >= 18) & (hour < 22))] = peak  # peak
    return np.round(price, 2)


def _build_frame(days: int, seed: int) -> pd.DataFrame:
    """Assemble the full hourly dataframe for `days` days from `seed`."""
    rng = np.random.default_rng(seed)
    periods = days * 24
    idx = pd.date_range(start=_START_DATE, periods=periods, freq="h")

    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    month = idx.month.to_numpy()
    dow = idx.dayofweek.to_numpy()            # 0 = Monday
    is_weekend = (dow >= 5).astype(int)

    # --- Weather -----------------------------------------------------------
    # Temperature: seasonal (peak ~ May) + diurnal (peak ~ 15:00) + noise.
    seasonal_t = 8.0 * np.sin(2 * np.pi * (doy - 105) / 365.0)   # +/-8 degC swing
    diurnal_t = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)      # +/-5 degC swing
    temperature_c = 28.0 + seasonal_t + diurnal_t + rng.normal(0, 1.0, periods)

    # Cloud cover [0,1]: wetter monsoon months (Jun-Sep) are cloudier.
    monsoon = np.isin(month, [6, 7, 8, 9])
    cloud_mean = np.where(monsoon, 0.55, 0.25)
    cloud_cover = np.clip(rng.beta(2.0, 4.0, periods) + (cloud_mean - 0.33), 0.0, 1.0)

    # Wind speed (m/s): Weibull, slightly windier in monsoon.
    wind_scale = np.where(monsoon, 7.5, 6.0)
    wind_speed_ms = np.round(wind_scale * rng.weibull(2.0, periods), 2)

    # --- Generation --------------------------------------------------------
    clear_sky = _CLEAR_SKY_PEAK_WM2 * _solar_elevation_sin(doy, hour)
    solar_irradiance_wm2 = np.round(clear_sky * (1.0 - 0.75 * cloud_cover), 1)
    solar_power_kw = np.round(
        _solar_power_kw(solar_irradiance_wm2, temperature_c), 3
    )
    wind_power_kw = np.round(_wind_power_kw(wind_speed_ms), 3)

    # --- Demand ------------------------------------------------------------
    # Two daily peaks (morning ~08:00, evening ~19:00) as gaussians on the hour.
    morning = 18.0 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening = 25.0 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    daily_shape = morning + evening
    weekend_factor = np.where(is_weekend == 1, 0.85, 1.0)       # lighter weekends
    summer_cooling = np.where(np.isin(month, [3, 4, 5, 6]), 8.0, 0.0)
    demand_kw = (
        _DEMAND_BASE_KW
        + daily_shape * weekend_factor
        + summer_cooling
        + rng.normal(0, _DEMAND_NOISE_KW, periods)
    )
    demand_kw = np.round(np.clip(demand_kw, 5.0, None), 3)

    # --- Price -------------------------------------------------------------
    price_inr_per_kwh = _time_of_use_price(hour)

    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "hour": hour,
            "day_of_week": dow,
            "month": month,
            "is_weekend": is_weekend,
            "temperature_c": np.round(temperature_c, 2),
            "cloud_cover": np.round(cloud_cover, 3),
            "wind_speed_ms": wind_speed_ms,
            "solar_irradiance_wm2": solar_irradiance_wm2,
            "solar_power_kw": solar_power_kw,
            "wind_power_kw": wind_power_kw,
            "demand_kw": demand_kw,
            "price_inr_per_kwh": price_inr_per_kwh,
        }
    )
    return frame


# Column -> (unit, description) for the data dictionary and downstream stages.
_DATA_DICTIONARY: list[tuple[str, str, str]] = [
    ("timestamp", "ISO datetime", "Local hour-beginning timestamp (hourly steps)."),
    ("hour", "0-23", "Hour of day - feature for the diurnal cycle."),
    ("day_of_week", "0-6", "Day of week (0 = Monday) - feature."),
    ("month", "1-12", "Calendar month - feature for seasonality."),
    ("is_weekend", "0/1", "1 on Saturday/Sunday - feature for demand pattern."),
    ("temperature_c", "degC", "Ambient temperature: seasonal + diurnal + noise."),
    ("cloud_cover", "0-1", "Fractional cloud cover (higher in monsoon)."),
    ("wind_speed_ms", "m/s", "Hub-height wind speed (Weibull distributed)."),
    ("solar_irradiance_wm2", "W/m^2", "Clear-sky irradiance attenuated by cloud."),
    ("solar_power_kw", "kW", "PV output = f(irradiance, temperature). TARGET."),
    ("wind_power_kw", "kW", "Turbine output via power curve = f(wind). TARGET."),
    ("demand_kw", "kW", "Site electrical demand/load. TARGET."),
    ("price_inr_per_kwh", "Rs/kWh", "Time-of-use grid import price. TARGET."),
]


def _write_data_dictionary(frame: pd.DataFrame, days: int, seed: int) -> None:
    """Write a Markdown data dictionary documenting columns and assumptions."""
    lines = [
        "# Data Dictionary - Microgrid_AI (Stage 1)",
        "",
        "> **SYNTHETIC data.** Generated by `datagen/generate.py` from documented",
        "> physical models - it is *not* measured field data. It is fully",
        f"> reproducible: `python main.py data --days {days} --seed {seed}`.",
        "",
        f"- Rows: **{len(frame):,}** hourly records "
        f"({days} days from {_START_DATE}).",
        f"- Random seed: **{seed}**",
        f"- Site latitude: **{config.SITE_LATITUDE_DEG} degN** (solar geometry).",
        "",
        "## Columns",
        "",
        "| Column | Unit | Description |",
        "| ------ | ---- | ----------- |",
    ]
    for name, unit, desc in _DATA_DICTIONARY:
        lines.append(f"| `{name}` | {unit} | {desc} |")

    lines += [
        "",
        "## Plant specification (from `config.py`)",
        "",
        f"- PV array rating: **{config.PV_RATED_KWP} kWp** "
        f"(performance ratio {_PV_PERFORMANCE_RATIO}).",
        f"- Wind turbine: **{config.WIND_RATED_KW} kW** rated; "
        f"cut-in {config.WIND_CUT_IN_MS}, rated {config.WIND_RATED_MS}, "
        f"cut-out {config.WIND_CUT_OUT_MS} m/s.",
        f"- Battery: **{config.BATTERY_CAPACITY_KWH} kWh**, "
        f"+/-{config.BATTERY_MAX_CHARGE_KW} kW, "
        f"SoC {int(config.BATTERY_MIN_SOC * 100)}-"
        f"{int(config.BATTERY_MAX_SOC * 100)} %, "
        f"round-trip eff {config.BATTERY_ROUND_TRIP_EFF}.",
        f"- Grid import (peak) price: **Rs {config.GRID_IMPORT_PRICE}/kWh**; "
        f"grid CO2 intensity **{config.CO2_PER_KWH_GRID} kg/kWh**.",
        "",
        "## Modelling assumptions",
        "",
        "1. **Solar** - clear-sky irradiance from declination + hour angle at the",
        "   site latitude, attenuated by `0.75 * cloud_cover`; converted to power",
        "   with the PV rating, performance ratio, and a -0.4 %/degC temperature",
        "   derate above 25 degC.",
        "2. **Wind** - Weibull(shape=2) wind speed through a cut-in/rated/cut-out",
        "   power curve (cubic ramp region).",
        "3. **Demand** - overnight base load plus morning and evening gaussian",
        "   peaks, scaled down 15 % on weekends, plus a summer cooling bump.",
        "4. **Price** - three-band time-of-use tariff (off-peak nights, peak",
        "   09:00-12:00 & 18:00-22:00, normal otherwise).",
        "5. **Battery SoC** is a *control* variable decided by the EMS in later",
        "   stages, so it is intentionally not a column in this dataset.",
        "",
    ]
    DATA_DICTIONARY_MD.write_text("\n".join(lines), encoding="utf-8")


def _sanity_check(frame: pd.DataFrame) -> None:
    """Fail loudly if the generated data violates basic physical expectations."""
    assert not frame.isnull().any().any(), "dataset contains NaNs"
    for col in ("solar_power_kw", "wind_power_kw", "demand_kw", "price_inr_per_kwh"):
        assert (frame[col] >= 0).all(), f"{col} has negative values"
    assert (frame["solar_power_kw"] <= config.PV_RATED_KWP + 1e-6).all(), \
        "solar exceeds PV rating"
    assert (frame["wind_power_kw"] <= config.WIND_RATED_KW + 1e-6).all(), \
        "wind exceeds turbine rating"
    # Solar must be zero at local midnight.
    assert frame.loc[frame["hour"] == 0, "solar_power_kw"].max() == 0.0, \
        "non-zero solar at midnight"


def generate_dataset(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate, validate, and persist the Stage 1 dataset + data dictionary.

    Args:
        days: number of days of hourly data to generate.
        seed: RNG seed for full reproducibility.

    Returns:
        The generated dataframe (also written to ``data/microgrid_dataset.csv``).
    """
    log.info("Generating synthetic dataset: %d days, seed=%d", days, seed)
    frame = _build_frame(days, seed)
    _sanity_check(frame)

    frame.to_csv(DATASET_CSV, index=False)
    _write_data_dictionary(frame, days, seed)

    log.info("Wrote %d rows -> %s", len(frame), DATASET_CSV)
    log.info("Wrote data dictionary -> %s", DATA_DICTIONARY_MD)
    _log_summary(frame)
    return frame


def _log_summary(frame: pd.DataFrame) -> None:
    """Log a short, honest summary of the generated series."""
    def _stat(col: str) -> str:
        s = frame[col]
        return f"mean={s.mean():.2f} min={s.min():.2f} max={s.max():.2f}"

    log.info("  solar_power_kw : %s", _stat("solar_power_kw"))
    log.info("  wind_power_kw  : %s", _stat("wind_power_kw"))
    log.info("  demand_kw      : %s", _stat("demand_kw"))
    log.info("  price          : %s", _stat("price_inr_per_kwh"))
    gen = frame["solar_power_kw"].sum() + frame["wind_power_kw"].sum()
    dem = frame["demand_kw"].sum()
    log.info(
        "  total renewable gen = %.0f kWh vs demand = %.0f kWh (ratio %.2f)",
        gen, dem, gen / dem,
    )
