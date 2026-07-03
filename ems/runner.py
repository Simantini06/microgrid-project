"""Stage 3 - run a single controller over the eval window and persist results.

Kept separate from any one approach so the Stage 6 comparison harness can reuse
it for the baseline, generative, and agentic controllers alike.
"""
from __future__ import annotations

import json

import pandas as pd

import config
from ems.environment import Controller, SimulationResult, load_eval_window, simulate
from ems.metrics import compute_metrics, format_metrics
from utils.logger import get_logger

log = get_logger("ems.runner")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def run_controller(
    controller: Controller,
    window: pd.DataFrame | None = None,
) -> tuple[SimulationResult, dict]:
    """Simulate one controller and return (raw result, metric dict)."""
    if window is None:
        window = load_eval_window()
    sim = simulate(window, controller)
    metrics = compute_metrics(sim)
    log.info("[%s] metrics:\n%s", controller.name, format_metrics(metrics))
    return sim, metrics


def save_run(sim: SimulationResult, metrics: dict) -> tuple:
    """Write the hourly log (CSV) and metrics (JSON) to reports/."""
    slug = _slug(sim.controller_name)
    hourly_path = config.REPORTS_DIR / f"run_{slug}_hourly.csv"
    metrics_path = config.REPORTS_DIR / f"run_{slug}_metrics.json"
    sim.hourly.to_csv(hourly_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved -> %s and %s", hourly_path.name, metrics_path.name)
    return hourly_path, metrics_path
