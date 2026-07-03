"""Central configuration for the Microgrid_AI project.

Loads environment variables (Groq API key, model names, domain constants) and
defines every filesystem path used across the build stages. Import values from
here instead of hard-coding paths or model names elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Base + .env ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Filesystem paths -------------------------------------------------------
DATASET_DIR = BASE_DIR / "dataset"     # raw / external data
DATA_DIR = BASE_DIR / "data"           # processed data + data dictionary
MODELS_DIR = BASE_DIR / "models"       # persisted ML models
REPORTS_DIR = BASE_DIR / "reports"     # generated reports
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

for _d in (DATASET_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
           STATIC_DIR, TEMPLATES_DIR):
    _d.mkdir(exist_ok=True)

# --- Groq / LLM configuration ----------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# Production Groq-hosted models (verified against console.groq.com/docs/models).
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST: str = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Model used by BOTH EMS controllers (generative + agentic) so the comparison
# is fair (same model, different paradigm). Defaults to the fast 8B model: the
# agentic loop makes several calls/hour, and the free Groq tier caps the 70B
# model at 100k tokens/day - too little for a full agentic week. Set
# EMS_LLM_MODEL=llama-3.3-70b-versatile in .env if you have paid quota.
EMS_LLM_MODEL: str = os.getenv("EMS_LLM_MODEL", GROQ_MODEL_FAST)

# --- Domain constants (India context: Rs and CO2) ---------------------------
GRID_IMPORT_PRICE: float = float(os.getenv("GRID_IMPORT_PRICE", "8.0"))  # Rs/kWh peak
CO2_PER_KWH_GRID: float = float(os.getenv("CO2_PER_KWH_GRID", "0.82"))   # kg CO2 / kWh

# --- Physical plant specification (single small commercial microgrid) -------
# These describe the simulated site and are reused by every stage (data
# generation, forecasting, baseline EMS, and both AI approaches).
SITE_LATITUDE_DEG: float = 13.08       # Chennai, India (drives solar geometry)
PV_RATED_KWP: float = 50.0             # rooftop PV array rating
WIND_RATED_KW: float = 20.0            # small wind turbine rated power
WIND_CUT_IN_MS: float = 3.0            # turbine starts generating
WIND_RATED_MS: float = 12.0            # turbine reaches rated power
WIND_CUT_OUT_MS: float = 25.0          # turbine shuts down (storm protection)

BATTERY_CAPACITY_KWH: float = 100.0    # usable battery energy capacity
BATTERY_MAX_CHARGE_KW: float = 25.0    # max charge power
BATTERY_MAX_DISCHARGE_KW: float = 25.0  # max discharge power
BATTERY_MIN_SOC: float = 0.20          # do not discharge below this (health)
BATTERY_MAX_SOC: float = 0.95          # do not charge above this (health)
BATTERY_ROUND_TRIP_EFF: float = 0.90   # round-trip efficiency (~0.95 each way)
BATTERY_INIT_SOC: float = 0.50         # state of charge at simulation start

# --- Grid tariff (import price is GRID_IMPORT_PRICE above) -------------------
GRID_EXPORT_PRICE: float = 3.0         # feed-in tariff (Rs/kWh) for exports

# --- Common evaluation window (same for baseline / LLM / agentic) -----------
# Friday -> Sunday so the window spans a weekday and a full weekend. Kept to
# 3 days so the agentic run (several Groq calls/hour, throttled by the free
# tier's tokens-per-minute limit) completes end-to-end in reasonable time.
EVAL_START_DATE: str = "2024-04-12"    # a Friday
EVAL_DAYS: int = 3                     # horizon all three approaches are scored on


def require_api_key() -> str:
    """Return the Groq API key, or raise a clear error if it is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
