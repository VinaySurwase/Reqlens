"""Idempotent ingestion + freshness lifecycle (spec 6.1).

Key invariants (all covered by tests/test_ingest.py):
  * same data twice => nothing changes except last_seen_at;
  * a posting is only counted missed after a run with status 'success';
  * partial/failed runs never expire anything;
  * a 'success' run that fetched 0 postings while active ones exist is demoted to 'partial'
    (an adapter returning nothing is far more likely a misconfiguration/outage than a mass delisting).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from reqlens.config import get_settings
from reqlens.db.models import IngestRun, RawPosting, Source
from reqlens.normalize import content_hash, normalize_company, strip_html
from reqlens.sources.base import RawPosting as FetchedPosting
from reqlens.sources.base import SourceAdapter

log = logging.getLogger("reqlens.ingest")


@dataclass
class IngestResult:
    run_id: int
    status: str
    fetched: int
    new: int
    changed: int
    unchanged: int
    expired: int
    errors: list[str]


def _get_or_create_source(session: Session, name: str) -> Source:
    src = session.scalar(select(Source).where(Source.name == name))
    if src is None:
        src = Source(name=name)
        session.add(src)
        session.flush()
    return src


async def run_ingest(
    session: Session, adapter: SourceAdapter, now: datetime | None = None, expire_after: int | None = None
) -> IngestResult:
    now = now or datetime.now(UTC)
    expire_after = expire_after if expire_after is not None else get_settings().expire_after_misses
    src = _get_or_create_source(session, adapter.name)
    run = IngestRun(source_id=src.id, started_at=now, status="failed")
    session.add(run)
    session.flush()

    existing: dict[str, RawPosting] = {
        r.source_job_id: r for r in session.scalars(select(RawPosting).where(RawPosting.source_id == src.id))
    }
    seen: set[str] = set()
    n_new = n_changed = n_same = fetched = 0
    fatal: str | None = None

    try:
        async for p in adapter.fetch():
            fetched += 1
            seen.add(p.source_job_id)
            ex = existing.get(p.source_job_id)
            outcome = _upsert(session, src.id, p, ex, now)
            if ex is None:
                existing[p.source_job_id] = outcome[1]
            if outcome[0] == "new":
                n_new += 1
            elif outcome[0] == "changed":
                n_changed += 1
            else:
                n_same += 1
    except Exception as e:  # noqa: BLE001 - any adapter failure is a failed/partial run, never a crash
        log.exception("adapter %s failed mid-fetch", adapter.name)
        fatal = repr(e)

    errors = list(getattr(adapter, "errors", []))
    if fatal:
        errors.append(fatal)
    had_active = any(r.status == "active" for r in existing.values() if r.source_job_id not in seen)
    if fatal:
        status = "partial" if fetched > 0 else "failed"
    elif errors:
        status = "partial"
    elif fetched == 0 and had_active:
        status = "partial"
        errors.append("fetched 0 postings while active postings exist; treated as partial, no expiry")
    else:
        status = "success"

    n_expired = 0
    if status == "success":
        for sjid, r in existing.items():
            if sjid in seen or r.status != "active":
                continue
            r.missed_runs += 1
            if r.missed_runs >= expire_after:
                r.status = "expired"
                r.dedup_dirty = True
                n_expired += 1
        src.last_success_at = now
    src.last_run_at = now

    run.finished_at = datetime.now(UTC)
    run.status = status
    run.fetched_count, run.new_count, run.changed_count = fetched, n_new, n_changed
    run.unchanged_count, run.expired_count = n_same, n_expired
    run.error_summary = "; ".join(errors)[:2000] or None
    session.commit()
    log.info(
        "ingest %s status=%s fetched=%d new=%d changed=%d unchanged=%d expired=%d",
        adapter.name, status, fetched, n_new, n_changed, n_same, n_expired,
    )
    return IngestResult(run.id, status, fetched, n_new, n_changed, n_same, n_expired, errors)


def _upsert(
    session: Session, source_id: int, p: FetchedPosting, ex: RawPosting | None, now: datetime
) -> tuple[str, RawPosting]:
    clean = strip_html(p.description_raw)
    h = content_hash(p.title, p.company, p.location_raw, clean)
    if ex is None:
        row = RawPosting(
            source_id=source_id, source_job_id=p.source_job_id, url=p.url, title=p.title, company=p.company,
            company_norm=normalize_company(p.company), location_raw=p.location_raw,
            description_raw=p.description_raw, description_clean=clean, posted_at=p.posted_at,
            content_hash=h, raw_payload=p.raw_payload, first_seen_at=now, last_seen_at=now,
            missed_runs=0, status="active", extraction_status="pending", dedup_dirty=True,
        )
        session.add(row)
        return "new", row
    reactivated = ex.status == "expired"
    ex.last_seen_at = now
    ex.missed_runs = 0
    if reactivated:
        ex.status = "active"
        ex.dedup_dirty = True
    if ex.content_hash == h:
        return "unchanged", ex
    ex.url, ex.title, ex.company, ex.company_norm = p.url, p.title, p.company, normalize_company(p.company)
    ex.location_raw, ex.description_raw, ex.description_clean = p.location_raw, p.description_raw, clean
    ex.posted_at = p.posted_at or ex.posted_at
    ex.content_hash, ex.raw_payload = h, p.raw_payload
    ex.last_changed_at = now
    ex.extraction_status, ex.extracted, ex.extraction_error = "pending", None, None
    ex.embedding = None
    ex.dedup_dirty = True
    return "changed", ex
