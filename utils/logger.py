"""Project-wide logging: formatted console output + rotating file handler.

Use `get_logger(__name__)` in any module. Handlers are configured once.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to the console and to logs/microgrid.log."""
    global _CONFIGURED
    if not _CONFIGURED:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(fmt)

        file_handler = RotatingFileHandler(
            LOGS_DIR / "microgrid.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(console)
        root.addHandler(file_handler)
        _CONFIGURED = True

    return logging.getLogger(name)
