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

for _d in (DATASET_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)

# --- Groq / LLM configuration ----------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# Production Groq-hosted models (verified against console.groq.com/docs/models).
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST: str = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Domain constants (India context: Rs and CO2) ---------------------------
GRID_IMPORT_PRICE: float = float(os.getenv("GRID_IMPORT_PRICE", "8.0"))  # Rs/kWh peak
CO2_PER_KWH_GRID: float = float(os.getenv("CO2_PER_KWH_GRID", "0.82"))   # kg CO2 / kWh


def require_api_key() -> str:
    """Return the Groq API key, or raise a clear error if it is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
