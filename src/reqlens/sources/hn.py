"""Hacker News 'Ask HN: Who is hiring?' via the Algolia HN API. Shape from memory; UNVERIFIED.

Top-level comments are free-text postings. Company/title are heuristically parsed from the first line
(the community convention is ``Company | Role | Location | Remote | ...``); anything that does not
parse stays in the description and is handled by LLM extraction.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from reqlens.normalize import strip_html
from reqlens.sources.base import RawPosting
from reqlens.sources.http import PoliteClient

SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
ITEM = "https://hn.algolia.com/api/v1/items/{id}"


def parse_first_line(text: str) -> tuple[str, str, str]:
    """Return (company, title, location) from a HN-style header line; empty strings if unknown."""
    first = text.split("\n", 1)[0]
    parts = [p.strip() for p in first.split("|")]
    if len(parts) < 2:
        return "", "", ""
    company = parts[0]
    title = ""
    location = ""
    for p in parts[1:]:
        low = p.lower()
        if not title and re.search(
            r"engineer|developer|scientist|designer|manager|intern|analyst|sre|devops|researcher|architect|lead", low
        ):
            title = p
        elif not location and re.search(r"remote|onsite|on-site|hybrid|,|\b[A-Z]{2}\b", p):
            location = p
    return company, title or (parts[1] if len(parts) > 1 else ""), location


class HNWhoIsHiringAdapter:
    name = "hn"

    def __init__(self, http: PoliteClient | None = None, months: int = 1, max_comments: int = 2000) -> None:
        self.http = http or PoliteClient()
        self.months = months
        self.max_comments = max_comments
        self.errors: list[str] = []

    async def _find_threads(self) -> list[str]:
        data = await self.http.get_json(
            SEARCH, {"tags": "story,author_whoishiring", "query": "Ask HN: Who is hiring?", "hitsPerPage": 10}
        )
        hits = [
            h for h in data.get("hits", [])
            if "who is hiring" in (h.get("title") or "").lower() and "wants to be hired" not in (h.get("title") or "").lower()
        ]
        hits.sort(key=lambda h: h.get("created_at_i", 0), reverse=True)
        return [str(h["objectID"]) for h in hits[: self.months]]

    async def fetch(self) -> AsyncIterator[RawPosting]:
        try:
            threads = await self._find_threads()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"hn:find_thread: {e!r}")
            return
        if not threads:
            self.errors.append("hn: no 'Who is hiring' thread found")
            return
        for tid in threads:
            try:
                tree = await self.http.get_json(ITEM.format(id=tid))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"hn:{tid}: {e!r}")
                continue
            now = datetime.now(UTC)
            n = 0
            for c in tree.get("children", []) or []:
                if n >= self.max_comments:
                    break
                body = c.get("text")
                if not body or c.get("id") is None:
                    continue  # deleted/dead comment
                text = strip_html(body)
                company, title, location = parse_first_line(text)
                created = c.get("created_at_i")
                n += 1
                yield RawPosting(
                    source=self.name,
                    source_job_id=str(c["id"]),
                    url=f"https://news.ycombinator.com/item?id={c['id']}",
                    title=title,
                    company=company,
                    location_raw=location,
                    description_raw=body,
                    posted_at=datetime.fromtimestamp(created, UTC) if isinstance(created, int) else None,
                    raw_payload={"id": c.get("id"), "author": c.get("author"), "thread": tid, "text": body},
                    fetched_at=now,
                )
