"""Greenhouse public job-board adapter.

Endpoint shape is from memory and UNVERIFIED in this repo's build environment -- see docs/SOURCES.md.
The parser is defensive: missing fields become empty strings; a board that errors is recorded in
``errors`` (-> run status 'partial') and does not abort other boards.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from reqlens.sources.base import RawPosting
from reqlens.sources.http import PoliteClient

BASE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def _dt(v: Any) -> datetime | None:
    if not v or not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


class GreenhouseAdapter:
    name = "greenhouse"

    def __init__(self, boards: list[dict[str, str]], http: PoliteClient | None = None) -> None:
        self.boards = boards
        self.http = http or PoliteClient()
        self.errors: list[str] = []

    async def fetch(self) -> AsyncIterator[RawPosting]:
        for b in self.boards:
            token, company = b["token"], b.get("company", b["token"])
            try:
                data = await self.http.get_json(BASE.format(token=token), {"content": "true"})
                jobs = data.get("jobs")
                if not isinstance(jobs, list):
                    raise ValueError("unexpected response: no 'jobs' list")
            except Exception as e:  # noqa: BLE001 - per-board isolation is deliberate
                self.errors.append(f"greenhouse:{token}: {e!r}")
                continue
            now = datetime.now(UTC)
            for j in jobs:
                if "id" not in j:
                    continue
                yield RawPosting(
                    source=self.name,
                    source_job_id=f"{token}:{j['id']}",
                    url=j.get("absolute_url", ""),
                    title=j.get("title", ""),
                    company=j.get("company_name") or company,
                    location_raw=(j.get("location") or {}).get("name", ""),
                    description_raw=j.get("content", ""),
                    posted_at=_dt(j.get("first_published") or j.get("updated_at")),
                    raw_payload=j,
                    fetched_at=now,
                )
