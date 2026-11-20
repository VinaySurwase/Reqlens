"""Resumable extraction worker. State lives in raw_postings.extraction_status, so stopping and
restarting loses nothing. Quota errors back off exponentially; if they persist the worker stops
cleanly (rows stay 'pending')."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from reqlens.config import get_settings
from reqlens.db.models import RawPosting
from reqlens.extract.extractor import Extractor
from reqlens.extract.schema import ExtractInput
from reqlens.llm import QuotaError, TransientLLMError
from reqlens.observability.logging import record_call

log = logging.getLogger("reqlens.extract")


@dataclass
class ExtractStats:
    done: int = 0
    failed: int = 0
    retried: int = 0
    attempted: int = 0
    stopped_reason: str | None = None  # None | quota | transient

    @property
    def validation_failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def run_extraction(
    session: Session, extractor: Extractor, limit: int | None = None, batch_size: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ExtractStats:
    s = get_settings()
    batch_size = batch_size or s.extraction_batch_size
    stats = ExtractStats()
    quota_strikes = 0
    while limit is None or stats.attempted < limit:
        n = batch_size if limit is None else min(batch_size, limit - stats.attempted)
        rows = session.scalars(
            select(RawPosting).where(RawPosting.extraction_status == "pending")
            .where(RawPosting.status == "active").order_by(RawPosting.id).limit(n)
        ).all()
        if not rows:
            break
        for r in rows:
            inp = ExtractInput(title=r.title, company=r.company, location=r.location_raw, description=r.description_clean)
            try:
                out = extractor.extract(inp)
            except QuotaError:
                quota_strikes += 1
                session.commit()
                if quota_strikes > s.llm_max_retries_on_quota:
                    stats.stopped_reason = "quota"
                    log.warning("extraction stopped: quota exhausted; %d rows remain pending", len(rows))
                    return stats
                sleep(min(s.llm_max_backoff_s, 2.0**quota_strikes))
                break  # re-query pending rows and continue (resumable by construction)
            except TransientLLMError as e:
                quota_strikes += 1
                session.commit()
                if quota_strikes > s.llm_max_retries_on_quota:
                    stats.stopped_reason = "transient"
                    log.warning("extraction stopped: transient errors persist: %s", e)
                    return stats
                sleep(min(s.llm_max_backoff_s, 2.0**quota_strikes))
                break
            quota_strikes = 0
            stats.attempted += 1
            for c in out.calls:
                record_call(session, "extract", extractor.model, c.usage.input_tokens, c.usage.output_tokens,
                            c.usage.latency_ms, c.ok)
            if out.retried:
                stats.retried += 1
            if out.result is not None:
                r.extracted = out.result.model_dump()
                r.extraction_status, r.extraction_error = "done", None
                stats.done += 1
            else:
                r.extracted, r.extraction_status, r.extraction_error = None, "failed", out.error
                stats.failed += 1
            r.dedup_dirty = True
        session.commit()
    session.commit()
    log.info("extraction done=%d failed=%d retried=%d validation_failure_rate=%.3f",
             stats.done, stats.failed, stats.retried, stats.validation_failure_rate)
    return stats
