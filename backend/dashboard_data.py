"""Stage 7 - assemble everything the dashboard renders.

Reads the Stage 6 comparison plus the per-approach hourly logs and builds: the
head-to-head metric table (with the best value flagged per row), a set of Plotly
figures (battery SoC, grid import, cumulative cost, and the headline savings /
cost-of-intelligence bars), and a sample of the LLM decision rationales. Figures
are serialised to JSON here and drawn client-side by Plotly in the template.

If the comparison has not been generated yet, `build_dashboard` returns a flag
the template uses to tell the user to run `python main.py compare` first.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import plotly.utils

import config
from backend.compare import _APPROACHES, _ROWS, _best_index

# One consistent colour per approach across every chart.
_COLORS = {
    "Rule-based": "#6c757d",
    "Generative-AI": "#0d6efd",
    "Agentic-AI": "#d63384",
}
_SLUG = {name: slug for name, slug in _APPROACHES}


def _fig_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _load_hourly() -> dict[str, pd.DataFrame]:
    """Load each approach's hourly log (skipping any that are missing)."""
    frames: dict[str, pd.DataFrame] = {}
    for name, slug in _APPROACHES:
        path = config.REPORTS_DIR / f"run_{slug}_hourly.csv"
        if path.exists():
            frames[name] = pd.read_csv(path, parse_dates=["timestamp"])
    return frames


def _line_figure(frames, column, title, ytitle) -> str:
    fig = go.Figure()
    for name, df in frames.items():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[column], mode="lines",
            name=name, line=dict(color=_COLORS.get(name), width=2),
        ))
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title=ytitle,
        margin=dict(l=50, r=20, t=50, b=40), height=360,
        legend=dict(orientation="h", y=-0.2), template="plotly_white",
    )
    return _fig_json(fig)


def _cumcost_figure(frames) -> str:
    fig = go.Figure()
    for name, df in frames.items():
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["cost_inr"].cumsum(), mode="lines",
            name=name, line=dict(color=_COLORS.get(name), width=2),
        ))
    fig.update_layout(
        title="Cumulative electricity cost (Rs)", xaxis_title="Time",
        yaxis_title="Rs (cumulative)", margin=dict(l=60, r=20, t=50, b=40),
        height=360, legend=dict(orientation="h", y=-0.2),
        template="plotly_white",
    )
    return _fig_json(fig)


def _bar_figure(results, key, title, ytitle) -> str:
    names = [n for n, _ in _APPROACHES]
    fig = go.Figure(go.Bar(
        x=names, y=[float(results[n][key]) for n in names],
        marker_color=[_COLORS[n] for n in names],
    ))
    fig.update_layout(
        title=title, yaxis_title=ytitle, margin=dict(l=60, r=20, t=50, b=40),
        height=320, template="plotly_white",
    )
    return _fig_json(fig)


def _metric_table(results) -> list[dict]:
    names = [n for n, _ in _APPROACHES]
    rows = []
    for label, key, better, fmt in _ROWS:
        values = [float(results[n][key]) for n in names]
        rows.append({
            "label": label,
            "cells": [fmt.format(v) for v in values],
            "best": _best_index(values, better),
        })
    return rows


def _rationales(frames, limit: int = 24) -> list[dict]:
    """Sample side-by-side decisions/rationales from the LLM approaches."""
    gen = frames.get("Generative-AI")
    ag = frames.get("Agentic-AI")
    if gen is None or ag is None:
        return []
    n = min(limit, len(gen), len(ag))
    out = []
    for i in range(n):
        g, a = gen.iloc[i], ag.iloc[i]
        out.append({
            "timestamp": str(g["timestamp"]),
            "price": f"{g['price_inr_per_kwh']:.1f}",
            "gen_batt": f"{g['battery_kw']:+.1f}",
            "gen_rat": str(g.get("rationale", "")),
            "ag_batt": f"{a['battery_kw']:+.1f}",
            "ag_rat": str(a.get("rationale", "")),
        })
    return out


def build_dashboard() -> dict:
    """Return the full context dict for the dashboard template."""
    comparison_path = config.REPORTS_DIR / "comparison.json"
    if not comparison_path.exists():
        return {"ready": False}

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    results = payload["results"]
    frames = _load_hourly()

    return {
        "ready": True,
        "window_hours": payload["window_hours"],
        "model": payload["model"],
        "start_date": config.EVAL_START_DATE,
        "approaches": [n for n, _ in _APPROACHES],
        "colors": _COLORS,
        "metric_rows": _metric_table(results),
        "fig_cost_saved": _bar_figure(
            results, "cost_saved_pct", "Cost saved vs grid-only (%)", "%"),
        "fig_co2_saved": _bar_figure(
            results, "co2_saved_kg", "CO2 saved vs grid-only (kg)", "kg"),
        "fig_tokens": _bar_figure(
            results, "llm_tokens", "LLM tokens used", "tokens"),
        "fig_soc": _line_figure(
            frames, "soc", "Battery state of charge", "SoC (fraction)"),
        "fig_import": _line_figure(
            frames, "grid_import_kw", "Grid import", "kW"),
        "fig_cumcost": _cumcost_figure(frames),
        "rationales": _rationales(frames),
    }
