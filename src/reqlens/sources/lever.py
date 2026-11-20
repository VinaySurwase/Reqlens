"""Lever public postings adapter. Shape from memory; UNVERIFIED (docs/SOURCES.md)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

from reqlens.sources.base import RawPosting
from reqlens.sources.http import PoliteClient

BASE = "https://api.lever.co/v0/postings/{company}"


class LeverAdapter:
    name = "lever"

    def __init__(self, boards: list[dict[str, str]], http: PoliteClient | None = None) -> None:
        self.boards = boards
        self.http = http or PoliteClient()
        self.errors: list[str] = []

    async def fetch(self) -> AsyncIterator[RawPosting]:
        for b in self.boards:
            token, company = b["token"], b.get("company", b["token"])
            try:
                data = await self.http.get_json(BASE.format(company=token), {"mode": "json"})
                if not isinstance(data, list):
                    raise ValueError("unexpected response: not a list")
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"lever:{token}: {e!r}")
                continue
            now = datetime.now(UTC)
            for j in data:
                if "id" not in j:
                    continue
                cats = j.get("categories") or {}
                loc = cats.get("location", "") or ""
                wt = j.get("workplaceType")
                if wt:
                    loc = f"{loc} ({wt})".strip()
                created = j.get("createdAt")
                posted = datetime.fromtimestamp(created / 1000, UTC) if isinstance(created, int | float) else None
                desc = j.get("description") or j.get("descriptionPlain") or ""
                extra = " ".join(
                    f"{lst.get('text', '')} {lst.get('content', '')}" for lst in (j.get("lists") or [])
                )
                yield RawPosting(
                    source=self.name,
                    source_job_id=f"{token}:{j['id']}",
                    url=j.get("hostedUrl", ""),
                    title=j.get("text", ""),
                    company=company,
                    location_raw=loc,
                    description_raw=f"{desc}\n{extra}".strip(),
                    posted_at=posted,
                    raw_payload=j,
                    fetched_at=now,
                )
