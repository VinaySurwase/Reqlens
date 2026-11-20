"""Embedding maintenance: embeds only jobs whose embedding is NULL (set when embedding_text_hash changed)."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from reqlens.config import get_settings
from reqlens.db.models import Job
from reqlens.index.text import embedding_text_for_job
from reqlens.llm import Embedder, QuotaError, TransientLLMError
from reqlens.observability.logging import record_call

log = logging.getLogger("reqlens.index")


@dataclass
class IndexStats:
    embedded: int = 0
    stopped_reason: str | None = None


def run_indexing(session: Session, embedder: Embedder, batch_size: int = 64,
                 sleep: Callable[[float], None] = time.sleep) -> IndexStats:
    s = get_settings()
    stats = IndexStats()
    strikes = 0
    while True:
        jobs = session.scalars(select(Job).where(Job.embedding.is_(None)).order_by(Job.id).limit(batch_size)).all()
        if not jobs:
            break
        t0 = time.perf_counter()
        try:
            vecs = embedder.embed([embedding_text_for_job(j) for j in jobs], kind="doc")
        except (QuotaError, TransientLLMError):
            strikes += 1
            if strikes > s.llm_max_retries_on_quota:
                stats.stopped_reason = "quota"
                break
            sleep(min(s.llm_max_backoff_s, 2.0**strikes))
            continue
        strikes = 0
        for j, v in zip(jobs, vecs, strict=True):
            j.embedding = v
        record_call(session, "embed", embedder.model, sum(len(embedding_text_for_job(j)) // 4 for j in jobs), 0,
                    int((time.perf_counter() - t0) * 1000))
        session.commit()
        stats.embedded += len(jobs)
    session.commit()
    log.info("indexing %s", stats)
    return stats
