"""Stage 4 - Groq chat-model factory (LangChain wrapper).

Single place that constructs a `ChatGroq` client so every generative/agentic
call uses the same provider, model, and temperature from `config`. The API key
is read from `config` (which loads it from `.env`) and never hard-coded.
"""
from __future__ import annotations

from langchain_groq import ChatGroq

import config


def get_chat_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatGroq:
    """Return a configured ChatGroq client (Groq is the only LLM provider)."""
    config.require_api_key()
    return ChatGroq(
        model=model or config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        api_key=config.GROQ_API_KEY,
    )
