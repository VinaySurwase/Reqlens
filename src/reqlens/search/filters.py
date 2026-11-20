from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import ColumnElement, and_, func, or_

from reqlens.db.models import Job

Seniority = Literal["intern", "junior", "mid", "senior", "staff_plus", "unknown"]
Remote = Literal["onsite", "hybrid", "remote", "unknown"]


class SearchFilters(BaseModel):
    """Unknown keys are REJECTED (extra=forbid): the agent/LLM cannot smuggle in arbitrary filters."""

    model_config = ConfigDict(extra="forbid")
    remote_type: list[Remote] | None = None
    seniority: list[Seniority] | None = None
    exclude_seniority: list[Seniority] | None = None  # unknown seniority is NOT excluded (reported as uncertain)
    max_min_years_experience: int | None = Field(default=None, ge=0, le=30)
    include_unknown_years: bool = True
    skills_any: list[str] | None = Field(default=None, max_length=15)
    skills_all: list[str] | None = Field(default=None, max_length=15)
    company: str | None = Field(default=None, max_length=100)
    posted_within_days: int | None = Field(default=None, ge=1, le=3650)
    is_internship: bool | None = None
    status: Literal["active", "expired"] = "active"


def build_conditions(f: SearchFilters, now: datetime | None = None) -> list[ColumnElement[bool]]:
    now = now or datetime.now(UTC)
    c: list[ColumnElement[bool]] = [Job.status == f.status]
    if f.remote_type:
        c.append(Job.remote_type.in_(f.remote_type))
    if f.seniority:
        c.append(Job.seniority.in_(f.seniority))
    if f.exclude_seniority:
        c.append(Job.seniority.not_in(f.exclude_seniority))
    if f.max_min_years_experience is not None:
        cond = Job.min_years_experience <= f.max_min_years_experience
        c.append(or_(cond, Job.min_years_experience.is_(None)) if f.include_unknown_years else cond)
    if f.skills_any:
        c.append(Job.skills.overlap([s.lower().strip() for s in f.skills_any]))
    if f.skills_all:
        c.append(Job.skills.contains([s.lower().strip() for s in f.skills_all]))
    if f.company:
        from reqlens.normalize import normalize_company
        c.append(Job.company_norm == normalize_company(f.company))
    if f.posted_within_days:
        cutoff = now - timedelta(days=f.posted_within_days)
        c.append(func.coalesce(Job.posted_at, Job.first_seen_at) >= cutoff)  # first_seen when no posted_at
    if f.is_internship is not None:
        c.append(Job.is_internship.is_(f.is_internship))
    return [and_(*c)] if c else []
