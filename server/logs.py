"""Structured logs: one JSON object per line on stderr (A2H_LOG_FORMAT=text for humans).

    log.info("job finished", extra={"fields": {"job": jid, "status": "succeeded"}})
"""

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
               "level": record.levelname.lower(), "logger": record.name, "msg": record.getMessage()}
        out.update(getattr(record, "fields", None) or {})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record):
        fields = getattr(record, "fields", None) or {}
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        text = f"{base} {extra}".rstrip()
        return text + ("\n" + self.formatException(record.exc_info) if record.exc_info else "")


def setup(fmt="json", level="INFO"):
    """Idempotent: re-running (tests, --reload) replaces the handler instead of stacking another."""
    root = logging.getLogger("a2h")
    root.setLevel(level)
    root.propagate = False
    for h in list(root.handlers):
        if getattr(h, "_a2h", False):
            root.removeHandler(h)
    h = logging.StreamHandler(sys.stderr)
    h._a2h = True
    h.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(h)
    return root
