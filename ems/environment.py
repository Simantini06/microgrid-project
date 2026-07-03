"""Stage 3 - microgrid environment: physics, controller contract, simulation.

The environment is deliberately controller-agnostic. A *controller* only chooses
a battery power set-point each hour (positive = charge, negative = discharge);
the environment enforces the physical limits, settles the energy balance on the
hour's ACTUAL generation/demand, evolves the battery state of charge (SoC), and
records every energy flow for the metrics layer.

Sign convention for the battery set-point (`battery_kw`):
    > 0  -> charge  (store energy)
    < 0  -> discharge (deliver energy)
    = 0  -> idle
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
from utils.logger import get_logger

log = get_logger("ems.env")

# One-way efficiency (applied on both charge and discharge -> round-trip).
_ONE_WAY_EFF = math.sqrt(config.BATTERY_ROUND_TRIP_EFF)


# --------------------------------------------------------------------------- #
# Controller contract
# --------------------------------------------------------------------------- #
@dataclass
class DecisionContext:
    """Everything a controller may look at before acting for one hour.

    Generation/demand fields are *forecasts* (what the controller would know in
    advance). Price is the published TOU tariff, so it is known exactly.
    """
    timestamp: str
    hour: int
    soc: float                         # battery state of charge, fraction 0-1
    price_inr_per_kwh: float
    solar_forecast_kw: float
    wind_forecast_kw: float
    demand_forecast_kw: float

    # Plant limits, surfaced so controllers (and LLM prompts) can reason.
    battery_capacity_kwh: float = config.BATTERY_CAPACITY_KWH
    max_charge_kw: float = config.BATTERY_MAX_CHARGE_KW
    max_discharge_kw: float = config.BATTERY_MAX_DISCHARGE_KW
    min_soc: float = config.BATTERY_MIN_SOC
    max_soc: float = config.BATTERY_MAX_SOC
    export_price_inr_per_kwh: float = config.GRID_EXPORT_PRICE

    @property
    def renewable_forecast_kw(self) -> float:
        return self.solar_forecast_kw + self.wind_forecast_kw

    @property
    def net_forecast_kw(self) -> float:
        """Forecast surplus (>0) or deficit (<0) before the battery acts."""
        return self.renewable_forecast_kw - self.demand_forecast_kw

    def price_band(self) -> str:
        """Coarse tariff band, handy for rule-based and LLM reasoning."""
        peak = config.GRID_IMPORT_PRICE
        if self.price_inr_per_kwh >= 0.95 * peak:
            return "peak"
        if self.price_inr_per_kwh <= 0.55 * peak:
            return "off-peak"
        return "normal"


@dataclass
class Decision:
    """A controller's action for one hour, plus optional accounting metadata."""
    battery_kw: float                  # requested set-point (env will clip it)
    rationale: str = ""                # human-readable reason (LLM/agentic)
    llm_calls: int = 0                 # Groq calls made to produce this decision
    llm_tokens: int = 0                # total tokens consumed


class Controller:
    """Base class: every approach implements `decide` returning a Decision."""

    name: str = "controller"

    def decide(self, ctx: DecisionContext) -> Decision:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Environment physics
# --------------------------------------------------------------------------- #
@dataclass
class HourResult:
    """Settled energy flows for a single hour (all in kW == kWh over 1 h)."""
    timestamp: str
    price_inr_per_kwh: float
    solar_kw: float
    wind_kw: float
    demand_kw: float
    battery_kw: float                  # clipped, signed (+charge / -discharge)
    soc: float                         # SoC at END of hour
    grid_import_kw: float
    grid_export_kw: float
    renewable_used_kw: float           # renewable consumed on-site (load+charge)
    cost_inr: float                    # import cost minus export credit
    co2_kg: float                      # emissions from grid import
    grid_only_cost_inr: float          # reference: all demand from grid
    grid_only_co2_kg: float
    rationale: str = ""
    llm_calls: int = 0
    llm_tokens: int = 0
    decide_ms: float = 0.0             # wall-clock to produce this decision


class MicrogridEnvironment:
    """Stateless physics: given SoC + actuals + a battery request, settle 1 h."""

    def apply(
        self,
        soc: float,
        solar_kw: float,
        wind_kw: float,
        demand_kw: float,
        price: float,
        battery_request_kw: float,
    ) -> HourResult:
        cap = config.BATTERY_CAPACITY_KWH
        renewable = solar_kw + wind_kw

        # --- Clip the battery request to rate + SoC limits -----------------
        charge_kw = discharge_kw = 0.0
        soc_next = soc
        if battery_request_kw > 0:  # charge
            headroom_kwh = (config.BATTERY_MAX_SOC - soc) * cap
            charge_kw = min(
                battery_request_kw,
                config.BATTERY_MAX_CHARGE_KW,
                headroom_kwh / _ONE_WAY_EFF if headroom_kwh > 0 else 0.0,
            )
            soc_next = soc + (charge_kw * _ONE_WAY_EFF) / cap
        elif battery_request_kw < 0:  # discharge
            available_kwh = (soc - config.BATTERY_MIN_SOC) * cap
            discharge_kw = min(
                -battery_request_kw,
                config.BATTERY_MAX_DISCHARGE_KW,
                available_kwh * _ONE_WAY_EFF if available_kwh > 0 else 0.0,
            )
            soc_next = soc - (discharge_kw / _ONE_WAY_EFF) / cap

        # --- Settle the energy balance on ACTUALS --------------------------
        renewable_to_load = min(renewable, demand_kw)
        surplus = renewable - renewable_to_load          # >= 0
        unmet = demand_kw - renewable_to_load            # >= 0

        renew_to_batt = min(charge_kw, surplus)
        grid_to_batt = charge_kw - renew_to_batt
        batt_to_load = min(discharge_kw, unmet)
        batt_to_export = discharge_kw - batt_to_load

        grid_import = unmet - batt_to_load + grid_to_batt
        grid_export = (surplus - renew_to_batt) + batt_to_export
        renewable_used = renewable_to_load + renew_to_batt

        cost = grid_import * price - grid_export * config.GRID_EXPORT_PRICE
        co2 = grid_import * config.CO2_PER_KWH_GRID

        return HourResult(
            timestamp="",  # filled by simulate()
            price_inr_per_kwh=price,
            solar_kw=solar_kw,
            wind_kw=wind_kw,
            demand_kw=demand_kw,
            battery_kw=charge_kw - discharge_kw,
            soc=soc_next,
            grid_import_kw=grid_import,
            grid_export_kw=grid_export,
            renewable_used_kw=renewable_used,
            cost_inr=cost,
            co2_kg=co2,
            grid_only_cost_inr=demand_kw * price,
            grid_only_co2_kg=demand_kw * config.CO2_PER_KWH_GRID,
        )


# --------------------------------------------------------------------------- #
# Simulation loop
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    controller_name: str
    hourly: pd.DataFrame               # one row per settled hour
    total_decision_time_s: float
    llm_calls: int
    llm_tokens: int
    extras: dict = field(default_factory=dict)


def _load_checkpoint(
    path: Path, window: pd.DataFrame, controller_name: str
) -> list[dict]:
    """Return already-computed hour rows if the checkpoint matches this window."""
    if path is None or not path.exists():
        return []
    prev = pd.read_csv(path)
    wins = window["timestamp"].astype(str).tolist()
    if len(prev) <= len(window) and \
            prev["timestamp"].astype(str).tolist() == wins[:len(prev)]:
        log.info("[%s] resuming from checkpoint: %d/%d hours done.",
                 controller_name, len(prev), len(window))
        return prev.to_dict("records")
    log.warning("[%s] checkpoint does not match window -> starting fresh.",
                controller_name)
    path.unlink()
    return []


def simulate(
    window: pd.DataFrame,
    controller: Controller,
    init_soc: float = config.BATTERY_INIT_SOC,
    checkpoint_path: Path | None = None,
) -> SimulationResult:
    """Run `controller` over `window` and return the per-hour settled log.

    `window` must contain the actual columns (solar_power_kw, wind_power_kw,
    demand_kw, price_inr_per_kwh) and the forecast columns produced by
    `forecast.predict.add_forecasts` (solar_power_kw_forecast, ...).

    If `checkpoint_path` is given, each settled hour is appended to it and a
    re-run resumes from where a previous (possibly interrupted) run stopped -
    important for the long, rate-limited agentic run. Totals are derived from the
    per-hour rows, so they stay correct across a resume.
    """
    env = MicrogridEnvironment()
    rows: list[dict] = _load_checkpoint(checkpoint_path, window, controller.name)
    start_idx = len(rows)
    soc = float(rows[-1]["soc"]) if rows else init_soc

    for i in range(start_idx, len(window)):
        r = window.iloc[i]
        ctx = DecisionContext(
            timestamp=str(r["timestamp"]),
            hour=int(r["hour"]),
            soc=soc,
            price_inr_per_kwh=float(r["price_inr_per_kwh"]),
            solar_forecast_kw=float(r["solar_power_kw_forecast"]),
            wind_forecast_kw=float(r["wind_power_kw_forecast"]),
            demand_forecast_kw=float(r["demand_kw_forecast"]),
        )

        t0 = time.perf_counter()
        decision = controller.decide(ctx)
        decide_ms = (time.perf_counter() - t0) * 1000.0

        result = env.apply(
            soc=soc,
            solar_kw=float(r["solar_power_kw"]),
            wind_kw=float(r["wind_power_kw"]),
            demand_kw=float(r["demand_kw"]),
            price=float(r["price_inr_per_kwh"]),
            battery_request_kw=decision.battery_kw,
        )
        result.timestamp = str(r["timestamp"])
        result.rationale = decision.rationale
        result.llm_calls = decision.llm_calls
        result.llm_tokens = decision.llm_tokens
        result.decide_ms = decide_ms

        soc = result.soc
        rows.append(result.__dict__)
        if checkpoint_path is not None:
            pd.DataFrame([result.__dict__]).to_csv(
                checkpoint_path, mode="a",
                header=not checkpoint_path.exists(), index=False,
            )

    hourly = pd.DataFrame(rows)
    total_calls = int(hourly["llm_calls"].sum())
    total_tokens = int(hourly["llm_tokens"].sum())
    total_decide_time = float(hourly["decide_ms"].sum()) / 1000.0
    log.info(
        "[%s] simulated %d h | LLM calls=%d tokens=%d | decide time=%.3fs",
        controller.name, len(hourly), total_calls, total_tokens,
        total_decide_time,
    )
    return SimulationResult(
        controller_name=controller.name,
        hourly=hourly,
        total_decision_time_s=total_decide_time,
        llm_calls=total_calls,
        llm_tokens=total_tokens,
    )


def load_eval_window(
    start_date: str = config.EVAL_START_DATE,
    days: int = config.EVAL_DAYS,
) -> pd.DataFrame:
    """Load the dataset, attach Stage-2 forecasts, and slice the eval window."""
    from forecast.predict import add_forecasts

    csv = config.DATA_DIR / "microgrid_dataset.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found. Run `python main.py data`.")
    df = pd.read_csv(csv, parse_dates=["timestamp"]).sort_values("timestamp")
    df = add_forecasts(df).dropna(subset=["solar_power_kw_forecast"])

    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=days)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    window = window.reset_index(drop=True)
    if window.empty:
        raise ValueError(f"No rows in window {start_date} +{days}d after warmup.")
    log.info("Eval window: %s -> %s (%d hours)",
             window["timestamp"].iloc[0], window["timestamp"].iloc[-1], len(window))
    return window
