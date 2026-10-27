"""Simple JSON logging helpers for spine runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def setup_json_logger(name: str = "spine", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        extra_payload = getattr(record, "payload", None)
        if extra_payload is not None:
            payload["payload"] = extra_payload
        return json.dumps(payload, ensure_ascii=False)
