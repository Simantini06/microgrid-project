"""Stage 5 - Agentic-AI EMS controller (LangGraph, closed-loop, tool-using).

Where the Generative-AI controller (Stage 4) answers from a single prompt, this
one *acts*: a LangGraph ReAct agent is given a tool that simulates the immediate
consequence of any candidate battery set-point against the real environment
physics. The agent explores a few candidates, reads back their cost / CO2 /
resulting-SoC, then autonomously commits the best action - no human in the loop.

It implements the same `Controller` contract and runs on the same evaluation
week, and it records the true cost of the agentic loop: every LLM call and token
across the tool-use rounds, plus per-decision latency. Candidates are evaluated
on the *forecast* state (what the agent knows); the environment still settles the
hour on actuals, exactly like the other two approaches.

Robustness: the tool loop is bounded (`recursion_limit`), and if the agent fails
to return a parseable action it falls back to a safe idle (0 kW) rather than
crashing the simulation.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import config
from ems.environment import (
    Controller,
    Decision,
    DecisionContext,
    MicrogridEnvironment,
)
from llm.client import get_chat_llm
from llm.generative import _build_human_message, _parse_decision
from utils.logger import get_logger

log = get_logger("agents.agentic")

# Bounds the ReAct tool-use loop -> caps LLM calls (hence cost) per hour.
_RECURSION_LIMIT = 8

_SYSTEM_PROMPT = (
    "You are an autonomous energy-management agent for a small microgrid with "
    "solar, wind, a battery, and a grid connection. Choose the battery set-point "
    "for THIS hour that minimises electricity cost and grid CO2 while respecting "
    "the battery limits.\n"
    "You have a tool, evaluate_battery_action(battery_kw), that returns the "
    "immediate consequence of a candidate set-point: grid import, cost, CO2, and "
    "the resulting state-of-charge. Use it to compare 2-3 sensible candidates "
    "(for example: charge with any surplus, discharge to cover a deficit at peak "
    "price, or stay idle) before deciding. battery_kw > 0 charges, < 0 "
    "discharges.\n"
    "When you have decided, reply with ONLY a JSON object, no tool call, no "
    'markdown: {"battery_kw": <number>, "rationale": "<one concise sentence>"}'
)


class AgenticController(Controller):
    """LangGraph ReAct agent that simulates candidates via a tool, then acts."""

    name = "Agentic-AI"

    def __init__(self, llm=None) -> None:
        self.llm = llm or get_chat_llm(config.EMS_LLM_MODEL)
        self._env = MicrogridEnvironment()
        # Holds the hour currently being decided so the tool can read it.
        self._ctx: DecisionContext | None = None
        self.agent = create_react_agent(
            self.llm, tools=[self._make_eval_tool()], prompt=_SYSTEM_PROMPT
        )

    def _make_eval_tool(self):
        """Build the what-if tool bound to this controller's current context."""
        controller = self

        @tool
        def evaluate_battery_action(battery_kw: float) -> str:
            """Simulate one candidate battery set-point for the current hour.

            Args:
                battery_kw: candidate set-point; positive charges the battery,
                    negative discharges it, 0 is idle.

            Returns:
                JSON with the immediate outcome: the actually-applied set-point
                (after clamping to limits), grid import/export (kW), cost (Rs),
                CO2 (kg), and the resulting state-of-charge (%).
            """
            ctx = controller._ctx
            res = controller._env.apply(
                soc=ctx.soc,
                solar_kw=ctx.solar_forecast_kw,
                wind_kw=ctx.wind_forecast_kw,
                demand_kw=ctx.demand_forecast_kw,
                price=ctx.price_inr_per_kwh,
                battery_request_kw=battery_kw,
            )
            clamped = abs(res.battery_kw - battery_kw) > 0.1
            return json.dumps({
                "requested_kw": round(battery_kw, 2),
                "applied_kw": round(res.battery_kw, 2),
                "was_clamped_to_limits": clamped,
                "grid_import_kw": round(res.grid_import_kw, 2),
                "grid_export_kw": round(res.grid_export_kw, 2),
                "cost_inr": round(res.cost_inr, 2),
                "co2_kg": round(res.co2_kg, 2),
                "resulting_soc_pct": round(res.soc * 100, 1),
            })

        return evaluate_battery_action

    @staticmethod
    def _account(messages: list) -> tuple[int, int]:
        """Sum LLM calls and tokens across every AIMessage the agent produced."""
        calls = tokens = 0
        for msg in messages:
            if isinstance(msg, AIMessage):
                calls += 1
                usage = getattr(msg, "usage_metadata", None) or {}
                tokens += int(usage.get("total_tokens", 0))
        return calls, tokens

    def decide(self, ctx: DecisionContext) -> Decision:
        self._ctx = ctx
        inputs = {"messages": [HumanMessage(content=_build_human_message(ctx))]}
        try:
            state = self.agent.invoke(
                inputs, config={"recursion_limit": _RECURSION_LIMIT}
            )
        except Exception as exc:  # recursion / rate-limit / API errors
            log.warning("Agent failed for %s: %s", ctx.timestamp, exc)
            return Decision(
                battery_kw=0.0,
                rationale="Agent error -> safe idle (0 kW).",
                llm_calls=1,
                llm_tokens=0,
            )

        messages = state["messages"]
        calls, tokens = self._account(messages)

        # Final decision is the last AIMessage with textual content (no tool call).
        parsed = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                parsed = _parse_decision(msg.content)
                if parsed is not None:
                    break

        if parsed is None:
            return Decision(
                battery_kw=0.0,
                rationale="Agent gave no parseable action -> safe idle (0 kW).",
                llm_calls=max(calls, 1),
                llm_tokens=tokens,
            )

        battery_kw, rationale = parsed
        battery_kw = max(-ctx.max_discharge_kw, min(ctx.max_charge_kw, battery_kw))
        return Decision(
            battery_kw=battery_kw,
            rationale=rationale or "(no rationale given)",
            llm_calls=calls,
            llm_tokens=tokens,
        )
