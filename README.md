# Generative AI vs Agentic AI for Microgrid Operation — A Comparative Analysis

A final-year CSE capstone that solves the **same** microgrid energy-management
problem three ways and compares them honestly:

1. **Rule-based baseline** — a transparent heuristic EMS (the comparison floor).
2. **Generative AI** (LangChain + Groq) — explains and recommends; human-in-the-loop.
3. **Agentic AI** (LangGraph multi-agent) — autonomously decides and acts; closed-loop.

Everything is **local, runnable, and reproducible**. The deliverable is a
results table + written analysis showing where agentic AI helps — and where it
does **not** beat the simpler approaches.

> **Build status:** Stage 0 (Scaffold) complete. Stages 1–10 in progress.

---

## Tech stack
Python · Pandas · NumPy · scikit-learn · XGBoost · Plotly · **Groq API** ·
LangChain · LangGraph · FastAPI · Bootstrap 5 · SQLite.

**LLM provider: Groq only.** Models (verified on
[Groq docs](https://console.groq.com/docs/models)):
`llama-3.3-70b-versatile` (reasoning/agents) and `llama-3.1-8b-instant` (fast).

## Comparison metrics (the spine of the project)
- Forecast accuracy: **MAE / RMSE** per target (solar, wind, demand, SoC)
- Daily **energy cost (₹)**, **% renewable utilisation**, **CO₂ saved**
- **LLM call count** and **token cost** per approach
- **Decision latency** (wall-clock) per approach

Evaluated across: decision autonomy, optimisation quality, explainability,
latency, LLM/token cost, reliability & failure modes, implementation complexity.

---

## Setup

```bash
# 1. create & activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. configure secrets
copy .env.example .env       # then edit .env and add your GROQ_API_KEY
```

## Run

```bash
python main.py --help        # list all stage commands
python main.py data          # Stage 1: generate dataset (default 365 days, seed 42)
python main.py data --days 90 --seed 7   # smaller reproducible dataset
```

Stage 1 writes `data/microgrid_dataset.csv` (8,760 hourly rows) and
`data/data_dictionary.md`. The data is **clearly-labelled synthetic** — built
from documented physical models (solar geometry, turbine power curve, TOU
tariff), not measured field data, and fully reproducible from the seed.

> Requires Python 3.10+ (developed on 3.10; 3.12+ recommended).

---

## Folder map
```
Microgrid_AI/
  main.py            # CLI entry point (one subcommand per stage)
  config.py          # paths, Groq model config, domain constants
  requirements.txt   # pinned dependencies
  .env.example       # template for secrets (copy to .env)
  forecast/          # Stage 2 — forecasting models
  agents/            # Stage 5 — LangGraph multi-agent system
  llm/               # Stage 4 — LangChain + Groq generative layer
  backend/           # Stage 6/7 — comparison harness + FastAPI
  frontend/          # Stage 8 — educational explanation website
  utils/             # logging, shared metrics, IO helpers
  data/              # processed data + data dictionary  (gitignored)
  dataset/           # raw / external data               (gitignored)
  models/            # persisted ML models               (gitignored)
  reports/           # generated reports                 (gitignored)
  static/ templates/ # dashboard assets (Stage 7)
  logs/              # run logs                           (gitignored)
```

## Build stages
- [x] **Stage 0** — Scaffold (structure, config, logging, CLI, README skeleton)
- [x] **Stage 1** — Data (synthetic hourly dataset + data dictionary, reproducible)
- [ ] Stage 2 — Forecasting (RF / GB / XGBoost; MAE/RMSE; persist best)
- [ ] Stage 3 — Rule-based baseline EMS
- [ ] Stage 4 — Generative AI (LangChain + Groq)
- [ ] Stage 5 — Agentic AI (LangGraph)
- [ ] Stage 6 — Comparison harness (results table + analysis)
- [ ] Stage 7 — Dashboard (FastAPI + Bootstrap 5 + Plotly)
- [ ] Stage 8 — Explanation website
- [ ] Stage 9 — Report generation (HTML / Markdown / PDF-ready)
- [ ] Stage 10 — README & polish

## References
IEEE references will be added in Stage 10 (real, verifiable sources only).
