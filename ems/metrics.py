"""Stage 3 - metrics computed from a simulation's per-hour log.

Turns the hourly settled flows into the project's headline comparison metrics.
Every approach (baseline / generative / agentic) is scored with this one
function so the numbers are directly comparable. Nothing here is hand-set; all
values are aggregated from the environment's bookkeeping.
"""
from __future__ import annotations

import config
from ems.environment import SimulationResult


def compute_metrics(sim: SimulationResult) -> dict:
    """Aggregate a SimulationResult into the comparison metric set."""
    h = sim.hourly
    hours = len(h)
    days = hours / 24.0

    demand_kwh = float(h["demand_kw"].sum())
    renewable_gen_kwh = float((h["solar_kw"] + h["wind_kw"]).sum())
    renewable_used_kwh = float(h["renewable_used_kw"].sum())
    import_kwh = float(h["grid_import_kw"].sum())
    export_kwh = float(h["grid_export_kw"].sum())

    cost_inr = float(h["cost_inr"].sum())
    co2_kg = float(h["co2_kg"].sum())
    grid_only_cost = float(h["grid_only_cost_inr"].sum())
    grid_only_co2 = float(h["grid_only_co2_kg"].sum())

    # % of generated renewable energy consumed on-site (load + battery charge).
    self_consumption = (
        100.0 * renewable_used_kwh / renewable_gen_kwh
        if renewable_gen_kwh > 0 else 0.0
    )
    # % of demand covered by on-site renewable (energy autonomy).
    renewable_share = (
        100.0 * renewable_used_kwh / demand_kwh if demand_kwh > 0 else 0.0
    )

    avg_latency_ms = 1000.0 * sim.total_decision_time_s / hours if hours else 0.0

    return {
        "controller": sim.controller_name,
        "hours": hours,
        "days": round(days, 2),
        # --- Cost --------------------------------------------------------
        "total_cost_inr": round(cost_inr, 2),
        "daily_cost_inr": round(cost_inr / days, 2) if days else 0.0,
        "grid_only_cost_inr": round(grid_only_cost, 2),
        "cost_saved_inr": round(grid_only_cost - cost_inr, 2),
        "cost_saved_pct": round(
            100.0 * (grid_only_cost - cost_inr) / grid_only_cost, 2
        ) if grid_only_cost else 0.0,
        # --- Renewables --------------------------------------------------
        "renewable_generated_kwh": round(renewable_gen_kwh, 2),
        "renewable_used_kwh": round(renewable_used_kwh, 2),
        "renewable_self_consumption_pct": round(self_consumption, 2),
        "renewable_share_of_demand_pct": round(renewable_share, 2),
        "grid_import_kwh": round(import_kwh, 2),
        "grid_export_kwh": round(export_kwh, 2),
        # --- Emissions ---------------------------------------------------
        "co2_kg": round(co2_kg, 2),
        "grid_only_co2_kg": round(grid_only_co2, 2),
        "co2_saved_kg": round(grid_only_co2 - co2_kg, 2),
        # --- Battery -----------------------------------------------------
        "final_soc_pct": round(100.0 * float(h["soc"].iloc[-1]), 1),
        "avg_soc_pct": round(100.0 * float(h["soc"].mean()), 1),
        # --- AI cost / latency -------------------------------------------
        "llm_calls": sim.llm_calls,
        "llm_tokens": sim.llm_tokens,
        "avg_decision_latency_ms": round(avg_latency_ms, 3),
        "total_decision_time_s": round(sim.total_decision_time_s, 3),
    }


def format_metrics(metrics: dict) -> str:
    """Return a compact human-readable block for logging/reports."""
    m = metrics
    return (
        f"  cost         : Rs {m['total_cost_inr']:.0f} "
        f"({m['daily_cost_inr']:.0f}/day)  "
        f"saved {m['cost_saved_pct']:.1f}% vs grid-only\n"
        f"  renewables   : {m['renewable_self_consumption_pct']:.1f}% self-used, "
        f"{m['renewable_share_of_demand_pct']:.1f}% of demand\n"
        f"  grid         : import {m['grid_import_kwh']:.0f} kWh, "
        f"export {m['grid_export_kwh']:.0f} kWh\n"
        f"  CO2          : {m['co2_kg']:.0f} kg  "
        f"(saved {m['co2_saved_kg']:.0f} kg vs grid-only)\n"
        f"  battery      : avg SoC {m['avg_soc_pct']:.0f}%, "
        f"final {m['final_soc_pct']:.0f}%\n"
        f"  AI cost      : {m['llm_calls']} LLM calls, {m['llm_tokens']} tokens, "
        f"{m['avg_decision_latency_ms']:.2f} ms/decision"
    )
