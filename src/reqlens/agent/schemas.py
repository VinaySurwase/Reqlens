from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from reqlens.search.filters import SearchFilters


class SearchJobsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = Field(default=10, ge=1, le=20)


class GetJobDetailsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: int = Field(ge=1)


TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "search_jobs",
     "description": ("Search active job postings. 'query' is free text for role/skills/topic. Put hard constraints "
                     "(remote_type, seniority, exclude_seniority, max_min_years_experience, skills_all/skills_any, "
                     "company, posted_within_days, is_internship) in 'filters'. Returns jobs with structured fields; "
                     "a null field means UNKNOWN, not 'no requirement'."),
     "parameters": SearchJobsArgs.model_json_schema()},
    {"name": "get_job_details",
     "description": "Get full details (description excerpt, all apply URLs) for one job_id returned by search_jobs.",
     "parameters": GetJobDetailsArgs.model_json_schema()},
]


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class RankedJob(BaseModel):
    job_id: int
    why: str = Field(max_length=500)
    matched_constraints: list[str] = Field(default_factory=list)
    unmet_or_uncertain: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    answer_summary: str = Field(max_length=1500)
    ranked: list[RankedJob] = Field(default_factory=list, max_length=20)


class AgentModel(Protocol):
    model: str

    def next_action(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ToolCall | None:
        """Return a ToolCall, or None when the model is done gathering information."""
        ...

    def finalize(self, history: list[dict[str, Any]]) -> AgentResult: ...


class FinalJob(BaseModel):
    job_id: int
    title: str
    company: str
    location: str
    apply_urls: list[dict[str, str]]
    why: str
    matched_constraints: list[str]
    unmet_or_uncertain: list[str]


class AgentResponse(BaseModel):
    status: Literal["ok", "partial", "failed"]
    answer_summary: str
    results: list[FinalJob]
    steps: int
    guard_events: list[str]
    trace: list[dict[str, Any]]
    latency_ms: int
