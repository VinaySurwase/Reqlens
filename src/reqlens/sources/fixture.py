"""Serve the synthetic fixtures through an httpx MockTransport, so the REAL adapters/parsers run offline."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import httpx

from reqlens.sources.http import PoliteClient

FIX = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


class FixtureServer:
    def __init__(self, directory: Path = FIX) -> None:
        self.data: dict[str, Any] = {n: json.loads((directory / f"{n}.json").read_text())
                                     for n in ("greenhouse", "lever", "hn", "remotive")}
        self.truth: dict[str, Any] = json.loads((directory / "truth.json").read_text())
        self.fail: set[str] = set()  # substrings of URLs that should return HTTP 500

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if any(f in url for f in self.fail):
            return httpx.Response(500, json={"error": "boom"})
        if m := re.search(r"greenhouse\.io/v1/boards/([^/]+)/jobs", url):
            d = self.data["greenhouse"].get(m.group(1))
            return httpx.Response(200, json=d) if d is not None else httpx.Response(404, json={})
        if m := re.search(r"api\.lever\.co/v0/postings/([^/?]+)", url):
            d = self.data["lever"].get(m.group(1))
            return httpx.Response(200, json=d) if d is not None else httpx.Response(404, json={})
        if "hn.algolia.com/api/v1/search_by_date" in url:
            return httpx.Response(200, json={"hits": [{"objectID": "9000001", "title": "Ask HN: Who is hiring? (October 2026)",
                                                       "created_at_i": 1790000000}]})
        if "hn.algolia.com/api/v1/items/" in url:
            return httpx.Response(200, json=self.data["hn"])
        if "remotive.com/api/remote-jobs" in url:
            return httpx.Response(200, json=self.data["remotive"])
        return httpx.Response(404, json={})

    def http(self) -> PoliteClient:
        return PoliteClient(httpx.AsyncClient(transport=httpx.MockTransport(self._handler)), use_cache=False)

    def adapters(self) -> dict[str, Any]:
        from reqlens.sources.greenhouse import GreenhouseAdapter
        from reqlens.sources.hn import HNWhoIsHiringAdapter
        from reqlens.sources.lever import LeverAdapter
        from reqlens.sources.remotive import RemotiveAdapter

        b = self.truth["boards"]
        return {
            "greenhouse": GreenhouseAdapter(b["greenhouse"], self.http()),
            "lever": LeverAdapter(b["lever"], self.http()),
            "hn": HNWhoIsHiringAdapter(self.http()),
            "remotive": RemotiveAdapter(self.http()),
        }
