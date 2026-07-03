"""Stage 7 - local FastAPI dashboard for the comparison.

Serves a single page (Bootstrap 5 + Plotly) that visualises the Stage 6
comparison: the metric table, the headline savings / cost-of-intelligence bars,
per-approach hourly plots (battery SoC, grid import, cumulative cost), and a
sample of the LLM decision rationales. Runs entirely locally - launch with
`python main.py dashboard` and open http://127.0.0.1:8000.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import config
from backend.dashboard_data import build_dashboard

app = FastAPI(title="Microgrid AI - Generative vs Agentic")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = build_dashboard()
    return templates.TemplateResponse(request, "dashboard.html", context)
