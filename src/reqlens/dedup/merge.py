"""Incremental dedup -> canonical jobs (spec 6.3 steps 5-6). Raw rows are never deleted; the link is
raw_postings.job_id, so a merge is reversible by re-running with a different threshold."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from reqlens.config import get_settings
from reqlens.db.models import Job, RawPosting, Source
from reqlens.dedup.score import DedupItem, Weights, cluster
from reqlens.index.text import build_embedding_text, embedding_hash
from reqlens.llm import Embedder, QuotaError, TransientLLMError
from reqlens.normalize import normalize_title

log = logging.getLogger("reqlens.dedup")

TSV_SQL = text(
    """UPDATE jobs SET tsv =
         setweight(to_tsvector('english', coalesce(title_display,'')), 'A') ||
         setweight(to_tsvector('english', coalesce(company_display,'')), 'B') ||
         setweight(to_tsvector('english', array_to_string(skills, ' ')), 'B') ||
         setweight(to_tsvector('english', coalesce(location_norm,'')), 'C') ||
         setweight(to_tsvector('english', left(coalesce(description_clean,''), 20000)), 'D')
       WHERE id = ANY(:ids)"""
)


@dataclass
class DedupStats:
    companies: int = 0
    jobs_created: int = 0
    jobs_updated: int = 0
    jobs_orphaned: int = 0
    stopped_reason: str | None = None


def weights_from_settings() -> Weights:
    s = get_settings()
    return Weights(s.dedup_w_title, s.dedup_w_desc, s.dedup_w_loc, s.dedup_w_seniority)


def completeness(r: RawPosting) -> int:
    e = r.extracted or {}
    return sum([
        e.get("seniority", "unknown") != "unknown", e.get("min_years_experience") is not None,
        e.get("remote_type", "unknown") != "unknown", bool(e.get("location_normalized")),
        bool(e.get("skills")), e.get("employment_type", "unknown") != "unknown",
    ])


def best_member(rows: list[RawPosting]) -> RawPosting:
    """Tie-break order: more complete extraction > longer description > earlier first_seen > lower id."""
    return max(rows, key=lambda r: (completeness(r), len(r.description_clean), -r.first_seen_at.timestamp(), -r.id))


def raw_embed_text(r: RawPosting) -> str:
    return f"{(r.extracted or {}).get('title_clean') or r.title}\n{r.description_clean[:1500]}"


def _to_item(r: RawPosting, source_names: dict[int, str] | None = None) -> DedupItem:
    emb = np.asarray(r.embedding, dtype=np.float32) if r.embedding is not None else None
    return DedupItem(r.id, r.title, r.location_raw, r.description_clean, r.extracted or {}, emb,
                     (source_names or {}).get(r.source_id, str(r.source_id)), r.source_job_id)


def canonicalize(session: Session, job: Job, members: list[RawPosting], source_names: dict[int, str]) -> None:
    best = best_member(members)
    e = best.extracted or {}
    active = [m for m in members if m.status == "active"]
    live = active or members
    urls: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for m in sorted(live, key=lambda m: m.id):
        if m.url and m.url not in seen_urls:
            seen_urls.add(m.url)
            urls.append({"source": source_names.get(m.source_id, str(m.source_id)), "url": m.url})
    job.company_norm = best.company_norm
    job.company_display = best.company
    job.title_display = e.get("title_clean") or best.title
    job.title_norm = normalize_title(job.title_display)
    job.location_norm = e.get("location_normalized", "")
    job.remote_type = e.get("remote_type", "unknown")
    job.seniority = e.get("seniority", "unknown")
    job.min_years_experience = e.get("min_years_experience")
    job.max_years_experience = e.get("max_years_experience")
    job.skills = list(e.get("skills", []))
    job.employment_type = e.get("employment_type", "unknown")
    job.is_internship = bool(e.get("is_internship", False))
    job.description_clean = best.description_clean
    job.apply_urls = urls
    posted = [m.posted_at for m in members if m.posted_at]
    job.posted_at = min(posted) if posted else None
    job.first_seen_at = min(m.first_seen_at for m in members)
    job.last_seen_at = max(m.last_seen_at for m in members)
    job.status = "active" if active else "expired"
    job.source_count = len({m.source_id for m in live})
    h = embedding_hash(build_embedding_text(
        job.title_display, job.company_display, job.location_norm, job.remote_type, job.seniority,
        list(job.skills), job.description_clean))
    if h != job.embedding_text_hash:
        job.embedding_text_hash = h
        job.embedding = None  # indexer re-embeds only rows whose text changed


def dedup_dirty(
    session: Session, embedder: Embedder | None, threshold: float | None = None, weights: Weights | None = None
) -> DedupStats:
    s = get_settings()
    threshold = s.dedup_threshold if threshold is None else threshold
    weights = weights or weights_from_settings()
    stats = DedupStats()
    companies = session.scalars(
        select(RawPosting.company_norm).where(RawPosting.dedup_dirty.is_(True))
        .where(RawPosting.extraction_status.in_(["done", "failed"])).distinct()
    ).all()
    touched_jobs: set[int] = set()
    source_names = {sid: name for sid, name in session.execute(select(Source.id, Source.name))}
    for comp in companies:
        rows = list(session.scalars(
            select(RawPosting).where(RawPosting.company_norm == comp).where(RawPosting.extraction_status == "done")
            .order_by(RawPosting.id)))
        failed_dirty = session.scalars(
            select(RawPosting).where(RawPosting.company_norm == comp)
            .where(RawPosting.extraction_status == "failed").where(RawPosting.dedup_dirty.is_(True))).all()
        stats.companies += 1
        if rows:
            if embedder is not None and len(rows) > 1 and comp != "":
                need = [r for r in rows if r.embedding is None]
                if need:
                    try:
                        vecs = embedder.embed([raw_embed_text(r) for r in need], kind="doc")
                    except (QuotaError, TransientLLMError) as e:
                        stats.stopped_reason = type(e).__name__
                        log.warning("dedup stopped for %r: %s", comp, e)
                        return stats
                    for r, v in zip(need, vecs, strict=True):
                        r.embedding = v
            items = [_to_item(r, source_names) for r in rows]
            groups = cluster(items, threshold, weights) if comp != "" else [[r.id] for r in rows]
            by_id = {r.id: r for r in rows}
            prior_jobs = {r.job_id for r in rows if r.job_id}
            for g in groups:
                mem = [by_id[i] for i in g]
                target_id = min((m.job_id for m in mem if m.job_id), default=None)
                if target_id is None:
                    job = Job(company_norm=comp, first_seen_at=datetime.now(UTC), last_seen_at=datetime.now(UTC))
                    session.add(job)
                    session.flush()
                    stats.jobs_created += 1
                else:
                    job = session.get(Job, target_id)  # type: ignore[assignment]
                    stats.jobs_updated += 1
                for m in mem:
                    m.job_id = job.id
                session.flush()
                canonicalize(session, job, mem, source_names)
                touched_jobs.add(job.id)
            session.flush()
            for jid in prior_jobs:  # jobs that lost every member (merged away) are retired, not deleted
                still = session.scalar(select(RawPosting.id).where(RawPosting.job_id == jid).limit(1))
                if still is None:
                    j = session.get(Job, jid)
                    if j is not None and (j.status != "expired" or j.source_count != 0):
                        j.status, j.source_count = "expired", 0
                        stats.jobs_orphaned += 1
                        touched_jobs.add(jid)
        for r in rows + list(failed_dirty):
            r.dedup_dirty = False
    if touched_jobs:
        session.execute(TSV_SQL, {"ids": list(touched_jobs)})
    session.commit()
    log.info("dedup %s", stats)
    return stats


def refresh_job_liveness(session: Session) -> int:
    """Cheap per-run sync of jobs.last_seen_at / status from member postings."""
    res = session.execute(text(
        """UPDATE jobs j SET last_seen_at = s.ls, status = s.st FROM (
             SELECT job_id, max(last_seen_at) ls,
                    CASE WHEN bool_or(status = 'active') THEN 'active' ELSE 'expired' END st
             FROM raw_postings WHERE job_id IS NOT NULL GROUP BY job_id) s
           WHERE j.id = s.job_id AND (j.last_seen_at IS DISTINCT FROM s.ls OR j.status IS DISTINCT FROM s.st)"""))
    session.commit()
    return getattr(res, "rowcount", 0) or 0


__all__ = ["dedup_dirty", "refresh_job_liveness", "canonicalize", "best_member", "DedupStats"]
