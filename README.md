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

Both the Generative-AI and Agentic-AI controllers use the **same** model
(`EMS_LLM_MODEL`, default `llama-3.1-8b-instant`) so the comparison is fair —
same model, different paradigm. The 8B default is deliberate: the agentic loop
makes several calls per hour, and Groq's free tier caps the 70B model at 100k
tokens/day (too little for a full agentic week), whereas the 8B model's daily
budget comfortably fits it. Set `EMS_LLM_MODEL=llama-3.3-70b-versatile` in `.env`
if you have paid Groq quota and want higher-quality reasoning.

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

```bash
python main.py forecast      # Stage 2: train solar/wind/demand forecasters
```

Stage 2 trains RandomForest / GradientBoosting / XGBoost per target on a
time-ordered hold-out split, compares them to a seasonal-naive baseline, and
persists the best model per target to `models/`. Metrics land in
`models/forecast_metrics.json` and `reports/forecast_metrics.md`. Forecasts use
calendar + lagged-actual features (realistic day-ahead framing — no perfect
future weather); wind is deliberately the hardest target.

```bash
python main.py baseline      # Stage 3: run the rule-based EMS over the eval week
```

Stage 3 introduces the shared **EMS environment** (`ems/`): battery physics,
energy balance, cost/CO₂/renewable accounting, and a common controller contract.
Controllers **decide on forecasts**; the environment **settles on actuals**. The
rule-based controller (no LLM) is the comparison floor — it writes
`reports/run_rule_based_metrics.json` and `reports/run_rule_based_hourly.csv`.
All three approaches are scored on the **same** evaluation week and metrics.

```bash
python main.py llm            # Stage 4: Generative-AI controller (calls Groq)
python main.py llm --hours 4  # cheap smoke test (first 4 hours only)
```

Stage 4 adds the **Generative-AI** controller (`llm/`): each hour the microgrid
state + forecasts are sent to a Groq LLM, which recommends a battery set-point
and explains it in one sentence (human-in-the-loop; single prompt, no tools). It
records the real cost of the LLM — **Groq call count, tokens, latency** — and
saves a `rationale` per hour. Requires `GROQ_API_KEY` in `.env`. Unparseable
replies fall back to a safe idle action, so a hiccup never crashes the run.

```bash
python main.py agentic            # Stage 5: Agentic-AI controller (Groq + tools)
python main.py agentic --hours 4  # cheap smoke test (a few hours only)
```

Stage 5 adds the **Agentic-AI** controller (`agents/`): a LangGraph ReAct agent
that, each hour, calls a `evaluate_battery_action` tool to simulate candidate
set-points against the real environment physics, reads back their cost / CO₂ /
resulting SoC, then **autonomously commits** the best action (closed-loop, no
human). It accounts for every LLM call and token across the tool-use rounds
(typically 2+ calls/hour). Same eval week and metrics as the other two.

```bash
python main.py compare           # Stage 6: build the head-to-head comparison
python main.py compare --rerun   # force fresh runs of all three approaches
```

Stage 6 assembles all three runs into one head-to-head table across every metric
and writes a grounded written analysis (`reports/comparison.md` +
`comparison.json`). By default it **reuses** the saved runs (so it never
re-spends Groq tokens); `--rerun` forces fresh runs. On the synthetic eval
window the honest finding is that the **rule-based baseline is the most
cost-effective**, the generative LLM's value is **explanation** rather than
better control, and the agentic approach's extra autonomy/compute **did not pay
off** — a deliberately nuanced result, not a foregone "AI wins".

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
  ems/               # Stage 3 — shared environment, metrics, rule-based EMS
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
- [x] **Stage 2** — Forecasting (RF / GB / XGBoost; MAE/RMSE vs naive; persist best)
- [x] **Stage 3** — Rule-based baseline EMS (shared environment + metrics)
- [x] **Stage 4** — Generative AI (LangChain + Groq; recommends + explains)
- [x] **Stage 5** — Agentic AI (LangGraph ReAct agent; simulates + acts)
- [x] **Stage 6** — Comparison harness (results table + written analysis)
- [ ] Stage 7 — Dashboard (FastAPI + Bootstrap 5 + Plotly)
- [ ] Stage 8 — Explanation website
- [ ] Stage 9 — Report generation (HTML / Markdown / PDF-ready)
- [ ] Stage 10 — README & polish

## References
IEEE references will be added in Stage 10 (real, verifiable sources only).
