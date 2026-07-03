"""Stage 4 - Generative-AI EMS controller (LangChain + Groq).

Each hour, the microgrid state and forecasts are rendered into a prompt and sent
to a Groq-hosted LLM, which *recommends* a battery set-point and *explains* it in
plain language. This is the "generative / human-in-the-loop" approach: the model
answers from a single prompt, it does not call tools or act on the grid itself.

It implements the same `Controller` contract as the rule-based baseline and runs
on the same evaluation week, and it records the real cost of using an LLM - Groq
call count, token usage, and per-decision latency - so the comparison is fair.

Robustness: the model is asked for strict JSON. If a response cannot be parsed
after a retry, the controller falls back to a safe "idle" action (0 kW) and says
so in the rationale, rather than crashing the simulation or inventing a number.
"""
from __future__ import annotations

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from ems.environment import Controller, Decision, DecisionContext
from llm.client import get_chat_llm
from utils.logger import get_logger

log = get_logger("llm.generative")

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0

_SYSTEM_PROMPT = (
    "You are an energy-management assistant for a small microgrid with rooftop "
    "solar, a wind turbine, a battery, and a grid connection. Each hour you "
    "recommend a single battery set-point to minimise electricity cost and grid "
    "CO2 while respecting the battery's limits.\n"
    "Conventions: battery_kw > 0 charges the battery, < 0 discharges it, 0 is "
    "idle. Charging during cheap/surplus hours and discharging during expensive "
    "(peak) hours lowers cost. Never exceed the stated charge/discharge limits "
    "or the min/max state-of-charge.\n"
    "Respond with ONLY a JSON object, no markdown and no code fences, exactly:\n"
    '{"battery_kw": <number>, "rationale": "<one concise sentence>"}'
)


def _build_human_message(ctx: DecisionContext) -> str:
    """Render the current hour's decision context as a compact prompt."""
    usable_kwh = (ctx.soc - ctx.min_soc) * ctx.battery_capacity_kwh
    headroom_kwh = (ctx.max_soc - ctx.soc) * ctx.battery_capacity_kwh
    return (
        f"Time: {ctx.timestamp} (hour {ctx.hour}).\n"
        f"Grid import price: Rs {ctx.price_inr_per_kwh:.2f}/kWh "
        f"({ctx.price_band()} tariff). Export/feed-in: "
        f"Rs {ctx.export_price_inr_per_kwh:.2f}/kWh.\n"
        f"Forecast this hour -> solar {ctx.solar_forecast_kw:.1f} kW, "
        f"wind {ctx.wind_forecast_kw:.1f} kW, demand {ctx.demand_forecast_kw:.1f} "
        f"kW (net {ctx.net_forecast_kw:+.1f} kW).\n"
        f"Battery: SoC {ctx.soc * 100:.0f}% of {ctx.battery_capacity_kwh:.0f} kWh; "
        f"can charge up to {ctx.max_charge_kw:.0f} kW "
        f"(room for ~{headroom_kwh:.0f} kWh), discharge up to "
        f"{ctx.max_discharge_kw:.0f} kW (~{usable_kwh:.0f} kWh available); "
        f"SoC must stay {ctx.min_soc * 100:.0f}-{ctx.max_soc * 100:.0f}%.\n"
        f"Recommend battery_kw for this hour."
    )


def _parse_decision(text: str) -> tuple[float, str] | None:
    """Extract (battery_kw, rationale) from a model reply, or None if invalid."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return float(obj["battery_kw"]), str(obj.get("rationale", "")).strip()
    except (ValueError, KeyError, TypeError):
        return None


class GenerativeController(Controller):
    """LLM recommends the battery action each hour (single-prompt, no tools)."""

    name = "Generative-AI"

    def __init__(self, llm=None) -> None:
        self.llm = llm or get_chat_llm()

    def decide(self, ctx: DecisionContext) -> Decision:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_human_message(ctx)),
        ]

        calls = 0
        tokens = 0
        parsed: tuple[float, str] | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.llm.invoke(messages)
            except Exception as exc:  # rate limits / transient API errors
                log.warning("Groq call failed (attempt %d/%d): %s",
                            attempt, _MAX_RETRIES, exc)
                time.sleep(_RETRY_BACKOFF_S * attempt)
                continue
            calls += 1
            usage = getattr(response, "usage_metadata", None) or {}
            tokens += int(usage.get("total_tokens", 0))
            parsed = _parse_decision(response.content or "")
            if parsed is not None:
                break
            log.warning("Unparseable reply (attempt %d/%d): %r",
                        attempt, _MAX_RETRIES, (response.content or "")[:120])

        if parsed is None:
            return Decision(
                battery_kw=0.0,
                rationale="LLM unavailable/unparseable -> safe idle (0 kW).",
                llm_calls=max(calls, 1),
                llm_tokens=tokens,
            )

        battery_kw, rationale = parsed
        # Clamp to physical rate limits (the environment also enforces this).
        battery_kw = max(-ctx.max_discharge_kw, min(ctx.max_charge_kw, battery_kw))
        return Decision(
            battery_kw=battery_kw,
            rationale=rationale or "(no rationale given)",
            llm_calls=calls,
            llm_tokens=tokens,
        )
