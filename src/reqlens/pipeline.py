"""ingest -> extract -> dedup -> index, each stage resumable and incremental."""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from sqlalchemy.orm import Session

from reqlens.config import Settings, get_settings
from reqlens.dedup.merge import dedup_dirty, refresh_job_liveness
from reqlens.extract.extractor import Extractor, GeminiExtractor
from reqlens.extract.heuristic import HeuristicExtractor
from reqlens.extract.worker import run_extraction
from reqlens.index.indexer import run_indexing
from reqlens.ingest.runner import run_ingest
from reqlens.llm import Embedder, GeminiClient
from reqlens.sources.base import SourceAdapter

log = logging.getLogger("reqlens.pipeline")


def build_extractor(settings: Settings | None = None) -> Extractor:
    s = settings or get_settings()
    if s.extraction_provider == "heuristic":
        return HeuristicExtractor()  # type: ignore[return-value]
    return GeminiExtractor(GeminiClient(s), s.max_description_chars)


async def run_pipeline(
    session: Session, adapters: list[SourceAdapter], extractor: Extractor, embedder: Embedder,
    dedup_threshold: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"ingest": {}}
    for a in adapters:
        r = await run_ingest(session, a)
        out["ingest"][a.name] = asdict(r)
    out["extract"] = asdict(run_extraction(session, extractor))
    out["dedup"] = asdict(dedup_dirty(session, embedder, dedup_threshold))
    out["liveness_updated"] = refresh_job_liveness(session)
    out["index"] = asdict(run_indexing(session, embedder))
    return out
