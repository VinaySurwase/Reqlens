"""Polite HTTP: descriptive User-Agent, timeouts, retry+backoff+jitter, on-disk response cache."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_none,
    wait_random_exponential,
)

from reqlens.config import get_settings


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class PoliteClient:
    def __init__(self, client: httpx.AsyncClient | None = None, use_cache: bool = True) -> None:
        s = get_settings()
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": s.user_agent}, timeout=s.http_timeout_s, follow_redirects=True
        )
        self._cache_dir = Path(s.http_cache_dir)
        self._ttl = s.http_cache_ttl_s
        self._use_cache = use_cache and s.http_cache_ttl_s > 0

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True, default=str).encode()).hexdigest()
        return self._cache_dir / f"{key}.json"

    async def _get(self, url: str, params: dict[str, Any] | None) -> Any:
        max_wait = get_settings().http_retry_max_wait_s
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(4),
            wait=wait_random_exponential(multiplier=1, max=max_wait) if max_wait > 0 else wait_none(),
            reraise=True,
        ):
            with attempt:
                r = await self._client.get(url, params=params)
                r.raise_for_status()
                return r.json()

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        path = self._cache_path(url, params)
        if self._use_cache and path.exists() and time.time() - path.stat().st_mtime < self._ttl:
            return json.loads(path.read_text())
        data = await self._get(url, params)
        if self._use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return data

    async def aclose(self) -> None:
        await self._client.aclose()
