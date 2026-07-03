"""Stage 3 - rule-based baseline EMS controller (the comparison floor).

A transparent, deterministic heuristic - NO LLM, no learning. It encodes the
kind of common-sense policy a human operator would write:

* **Surplus** (renewables forecast to exceed demand): store the surplus in the
  battery; whatever will not fit is exported.
* **Deficit** at **peak** tariff: discharge the battery to avoid buying
  expensive grid power.
* **Deficit** at **off-peak** tariff: keep the battery for the coming peak; if
  it is low, top it up cheaply from the grid.
* **Deficit** at **normal** tariff: discharge only if the battery is
  comfortably charged, otherwise ride on the grid.

Because it is rule-based, it makes zero LLM calls and decides in microseconds -
that is exactly the cost/latency floor the AI approaches must justify beating.
"""
from __future__ import annotations

import config
from ems.environment import Controller, Decision, DecisionContext

# SoC thresholds used by the heuristic (fractions of capacity).
_OFF_PEAK_TARGET_SOC = 0.80    # top up to this cheaply overnight
_NORMAL_DISCHARGE_SOC = 0.50   # only shave at normal tariff above this


class RuleBasedController(Controller):
    """Deterministic, explainable charge/discharge heuristic."""

    name = "Rule-based"

    def decide(self, ctx: DecisionContext) -> Decision:
        net = ctx.net_forecast_kw          # >0 surplus, <0 deficit
        band = ctx.price_band()

        if net > 0:
            # Surplus: charge with it (env caps to rate/SoC; rest is exported).
            return Decision(
                battery_kw=min(net, ctx.max_charge_kw),
                rationale=(
                    f"Surplus {net:.1f} kW forecast -> charge battery; "
                    f"export any excess."
                ),
            )

        deficit = -net
        if band == "peak":
            return Decision(
                battery_kw=-min(deficit, ctx.max_discharge_kw),
                rationale=(
                    f"Deficit {deficit:.1f} kW at PEAK tariff "
                    f"(Rs {ctx.price_inr_per_kwh:.1f}) -> discharge to avoid "
                    f"costly import."
                ),
            )

        if band == "off-peak":
            if ctx.soc < _OFF_PEAK_TARGET_SOC:
                headroom_kw = (_OFF_PEAK_TARGET_SOC - ctx.soc) \
                    * ctx.battery_capacity_kwh
                return Decision(
                    battery_kw=min(ctx.max_charge_kw, headroom_kw),
                    rationale=(
                        f"Deficit at OFF-PEAK tariff and SoC "
                        f"{ctx.soc * 100:.0f}% < {int(_OFF_PEAK_TARGET_SOC * 100)}"
                        f"% -> charge cheaply from grid for the coming peak."
                    ),
                )
            return Decision(
                battery_kw=0.0,
                rationale=(
                    "Deficit at OFF-PEAK tariff but battery already reserved "
                    "-> ride on cheap grid, save battery for peak."
                ),
            )

        # Normal tariff: shave with the battery only if it is well charged.
        if ctx.soc > _NORMAL_DISCHARGE_SOC:
            return Decision(
                battery_kw=-min(deficit, ctx.max_discharge_kw),
                rationale=(
                    f"Deficit {deficit:.1f} kW at normal tariff and SoC "
                    f"{ctx.soc * 100:.0f}% healthy -> discharge to shave import."
                ),
            )
        return Decision(
            battery_kw=0.0,
            rationale=(
                "Deficit at normal tariff with low SoC -> import from grid, "
                "preserve battery."
            ),
        )
