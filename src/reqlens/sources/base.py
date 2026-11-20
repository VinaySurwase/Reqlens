from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


class RawPosting(BaseModel):
    source: str
    source_job_id: str
    url: str = ""
    title: str = ""
    company: str = ""
    location_raw: str = ""
    description_raw: str = ""  # HTML or plain text
    posted_at: datetime | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime


class SourceAdapter(Protocol):
    name: str
    #: Human-readable problems hit during fetch(). Empty list + no exception == complete run.
    errors: list[str]

    def fetch(self) -> AsyncIterator[RawPosting]: ...
