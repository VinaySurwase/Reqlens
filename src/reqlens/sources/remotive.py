"""Remotive public API adapter. Shape from memory; UNVERIFIED. Their terms ask for attribution and
infrequent polling -- see docs/SOURCES.md; the 1h response cache plus 12h cron keeps usage low."""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

from reqlens.sources.base import RawPosting
from reqlens.sources.http import PoliteClient

URL = "https://remotive.com/api/remote-jobs"


class RemotiveAdapter:
    name = "remotive"

    def __init__(self, http: PoliteClient | None = None, category: str = "software-dev") -> None:
        self.http = http or PoliteClient()
        self.category = category
        self.errors: list[str] = []

    async def fetch(self) -> AsyncIterator[RawPosting]:
        try:
            data = await self.http.get_json(URL, {"category": self.category})
            jobs = data.get("jobs")
            if not isinstance(jobs, list):
                raise ValueError("unexpected response: no 'jobs' list")
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"remotive: {e!r}")
            return
        now = datetime.now(UTC)
        for j in jobs:
            if "id" not in j:
                continue
            posted = None
            if j.get("publication_date"):
                try:
                    posted = datetime.fromisoformat(str(j["publication_date"]).replace("Z", "+00:00"))
                    if posted.tzinfo is None:
                        posted = posted.replace(tzinfo=UTC)
                except ValueError:
                    posted = None
            yield RawPosting(
                source=self.name,
                source_job_id=str(j["id"]),
                url=j.get("url", ""),
                title=j.get("title", ""),
                company=j.get("company_name", ""),
                location_raw=f"Remote - {j.get('candidate_required_location', '')}".strip(" -"),
                description_raw=j.get("description", ""),
                posted_at=posted,
                raw_payload=j,
                fetched_at=now,
            )
