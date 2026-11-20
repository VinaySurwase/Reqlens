from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from reqlens.config import get_settings

EMBED_DIM = get_settings().embed_dim


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IngestRun(Base):
    __tablename__ = "ingest_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="failed")  # success|partial|failed
    fetched_count: Mapped[int] = mapped_column(Integer, default=0)
    new_count: Mapped[int] = mapped_column(Integer, default=0)
    changed_count: Mapped[int] = mapped_column(Integer, default=0)
    unchanged_count: Mapped[int] = mapped_column(Integer, default=0)
    expired_count: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text)


class RawPosting(Base):
    __tablename__ = "raw_postings"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    source_job_id: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(Text, default="")
    company: Mapped[str] = mapped_column(Text, default="")
    company_norm: Mapped[str] = mapped_column(String(255), default="")
    location_raw: Mapped[str] = mapped_column(Text, default="")
    description_raw: Mapped[str] = mapped_column(Text, default="")
    description_clean: Mapped[str] = mapped_column(Text, default="")
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64))
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    missed_runs: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|expired
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    extraction_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|done|failed
    extracted: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extraction_error: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBED_DIM), nullable=True)  # dedup only
    dedup_dirty: Mapped[bool] = mapped_column(default=True)

    __table_args__ = (
        UniqueConstraint("source_id", "source_job_id", name="uq_raw_source_job"),
        Index("ix_raw_company_status", "company_norm", "status"),
        Index("ix_raw_extraction_status", "extraction_status"),
    )


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_norm: Mapped[str] = mapped_column(String(255))
    company_display: Mapped[str] = mapped_column(Text, default="")
    title_norm: Mapped[str] = mapped_column(Text, default="")
    title_display: Mapped[str] = mapped_column(Text, default="")
    location_norm: Mapped[str] = mapped_column(Text, default="")
    remote_type: Mapped[str] = mapped_column(String(16), default="unknown")
    seniority: Mapped[str] = mapped_column(String(16), default="unknown")
    min_years_experience: Mapped[int | None] = mapped_column(Integer)
    max_years_experience: Mapped[int | None] = mapped_column(Integer)
    skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    employment_type: Mapped[str] = mapped_column(String(24), default="unknown")
    is_internship: Mapped[bool] = mapped_column(default=False)
    description_clean: Mapped[str] = mapped_column(Text, default="")
    apply_urls: Mapped[list[dict[str, str]]] = mapped_column(JSONB, default=list)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="active")
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    embedding_text_hash: Mapped[str] = mapped_column(String(64), default="")
    tsv: Mapped[Any | None] = mapped_column(TSVECTOR)
    source_count: Mapped[int] = mapped_column(Integer, default=1)

    __table_args__ = (Index("ix_jobs_company_status", "company_norm", "status"),)


class LlmCall(Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    purpose: Mapped[str] = mapped_column(String(16))  # extract|rerank|agent|judge|embed
    model: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd_estimate: Mapped[float] = mapped_column(default=0.0)  # free tier -> 0, tokens still recorded
    ok: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
