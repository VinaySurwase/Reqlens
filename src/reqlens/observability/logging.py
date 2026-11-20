from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from reqlens.db.models import LlmCall

run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")


def new_run_id() -> str:
    rid = uuid.uuid4().hex[:12]
    run_id.set(rid)
    return rid


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": run_id.get(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "event", None)
        if extra:
            d["event"] = extra
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)[-1500:]
        return json.dumps(d, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # never log URLs w/ query strings/keys


def record_call(
    session: Session, purpose: str, model: str, input_tokens: int, output_tokens: int, latency_ms: int,
    ok: bool = True,
) -> None:
    # Free tier => cost 0; tokens are still recorded for the eval's cost/latency columns.
    session.add(LlmCall(purpose=purpose, model=model[:64], input_tokens=input_tokens,
                        output_tokens=output_tokens, latency_ms=latency_ms, cost_usd_estimate=0.0, ok=ok))


class Timer:
    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a: object) -> None:
        self.ms = int((time.perf_counter() - self.t0) * 1000)
