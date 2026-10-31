"""Logging with secret redaction.

A filter scrubs bearer tokens and any value of known secret env vars from log
records, so that an accidental ``log.info(headers)`` cannot leak the Meta key.
"""

from __future__ import annotations

import logging
import os
import re

_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)
_SECRET_ENV = ("MODEL_API_KEY", "META_API_KEY")


def redact(text: str) -> str:
    text = _BEARER.sub(r"\1[REDACTED]", text)
    for name in _SECRET_ENV:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:
            return True
        record.msg = redact(msg)
        record.args = ()
        return True


def setup_logging(level: str | None = None) -> None:
    level = level or os.environ.get("MAANA_LOG_LEVEL", "INFO")
    root = logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for h in root.handlers for f in h.filters):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)
    root.setLevel(level)
    # httpx logs full URLs at INFO; keep it quieter
    logging.getLogger("httpx").setLevel(logging.WARNING)
