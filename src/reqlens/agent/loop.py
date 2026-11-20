"""Bounded tool-using agent (spec 6.6). The model never touches the DB: it can only call two validated
tools. Everything user-visible about a job (title, company, URLs) is joined server-side from the DB by
job_id; model prose is limited to 'why' and constraint lists, which are post-checked."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from reqlens.agent.schemas import (
    TOOL_SPECS,
    AgentModel,
    AgentResponse,
    AgentResult,
    FinalJob,
    GetJobDetailsArgs,
    SearchJobsArgs,
    ToolCall,
)
from reqlens.config import get_settings
from reqlens.db.models import Job
from reqlens.llm import Embedder, LLMClient, QuotaError, TransientLLMError
from reqlens.search.engine import job_detail, search

log = logging.getLogger("reqlens.agent")

# constraint-text keywords -> the structured field that must be known to claim the constraint is met
_FIELD_KEYWORDS = {
    "min_years_experience": re.compile(r"\b(years?|yrs?|experience)\b", re.I),
    "seniority": re.compile(r"\b(senior|junior|mid|staff|entry|level|intern)\b", re.I),
    "remote_type": re.compile(r"\b(remote|hybrid|on-?site|office)\b", re.I),
}


def _unknown_fields(j: Job) -> set[str]:
    out = set()
    if j.min_years_experience is None:
        out.add("min_years_experience")
    if j.seniority == "unknown":
        out.add("seniority")
    if j.remote_type == "unknown":
        out.add("remote_type")
    return out


def uncertainty_guard(j: Job, matched: list[str], uncertain: list[str]) -> tuple[list[str], list[str], int]:
    """A constraint cannot be 'matched' on a field whose value is unknown; move it to uncertain."""
    unknown = _unknown_fields(j)
    keep, moved = [], 0
    unc = list(uncertain)
    for c in matched:
        if any(f in unknown and rx.search(c) for f, rx in _FIELD_KEYWORDS.items()):
            unc.append(f"{c} (not verified: field unknown in posting)")
            moved += 1
        else:
            keep.append(c)
    return keep, unc, moved


def _exec_tool(session: Session, call: ToolCall, embedder: Embedder | None, llm: LLMClient | None,
               seen: dict[int, dict[str, Any]]) -> dict[str, Any]:
    try:
        if call.name == "search_jobs":
            a = SearchJobsArgs.model_validate(call.args)
            resp = search(session, a.query, "hybrid", a.k, a.filters, embedder, llm)
            rows = []
            for h in resp.results:
                row = {
                    "job_id": h.job_id,
                    "title": h.title,
                    "company": h.company,
                    "location": h.location,
                    "remote_type": h.remote_type,
                    "seniority": h.seniority,
                    "min_years_experience": h.min_years_experience,
                    "skills": h.skills[:8],
                    "sources": h.source_count,
                }
                rows.append(row)
                seen[h.job_id] = row
            return {"results": rows, "degraded": resp.degraded}
        if call.name == "get_job_details":
            a2 = GetJobDetailsArgs.model_validate(call.args)
            d = job_detail(session, a2.job_id)
            if d is None:
                return {"error": f"job_id {a2.job_id} not found"}
            seen.setdefault(a2.job_id, {"job_id": a2.job_id})
            return {"job_id": d["id"], "title": d["title_display"], "company": d["company_display"],
                    "description_excerpt_UNTRUSTED": (d["description_clean"] or "")[:1200],
                    "apply_urls": d["apply_urls"], "min_years_experience": d["min_years_experience"],
                    "seniority": d["seniority"], "remote_type": d["remote_type"]}
        return {"error": f"unknown tool {call.name!r}; available: search_jobs, get_job_details"}
    except ValidationError as e:
        return {"error": "invalid arguments: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])}
    except (QuotaError, TransientLLMError) as e:
        return {"error": f"temporarily unavailable: {type(e).__name__}"}


def run_agent(session: Session, query: str, model: AgentModel, embedder: Embedder | None, llm: LLMClient | None = None,
              ) -> AgentResponse:
    s = get_settings()
    t0 = time.perf_counter()
    history: list[dict[str, Any]] = [{"role": "user", "text": query[: s.max_query_chars]}]
    trace: list[dict[str, Any]] = []
    guard: list[str] = []
    seen: dict[int, dict[str, Any]] = {}
    last_search_order: list[int] = []
    steps = 0
    status = "ok"

    def over_budget() -> bool:
        if time.perf_counter() - t0 > s.agent_timeout_s:
            return True
        return sum(len(json.dumps(h, default=str)) for h in history) // 4 > s.agent_max_tokens

    try:
        while steps < s.max_tool_steps:
            if over_budget():
                guard.append("budget_exceeded")
                status = "partial"
                break
            call = model.next_action(history, TOOL_SPECS)
            if call is None:
                break
            steps += 1
            result = _exec_tool(session, call, embedder, llm, seen)
            if call.name == "search_jobs" and "results" in result:
                last_search_order = [r["job_id"] for r in result["results"]]
            history.append({"role": "model", "tool_call": call.model_dump()})
            history.append({"role": "tool", "name": call.name, "response": result})
            trace.append({"step": steps, "tool": call.name, "args": call.args,
                          "ok": "error" not in result, "n_results": len(result.get("results", []))})
        else:
            guard.append("max_steps_reached")
            status = "partial"
        final: AgentResult | None
        try:
            final = model.finalize(history)
        except (ValidationError, QuotaError, TransientLLMError, ValueError) as e:
            guard.append(f"finalize_failed:{type(e).__name__}")
            final, status = None, "partial"
    except (QuotaError, TransientLLMError) as e:
        guard.append(f"model_unavailable:{type(e).__name__}")
        final, status = None, "partial"

    ids_kept: list[int] = []
    out: list[FinalJob] = []
    if final is not None:
        for rj in final.ranked:
            if rj.job_id not in seen:
                guard.append(f"dropped_unseen_job_id:{rj.job_id}")  # hallucinated or never retrieved
                continue
            if rj.job_id in ids_kept:
                continue
            j = session.get(Job, rj.job_id)
            if j is None or j.status != "active":
                guard.append(f"dropped_inactive_job_id:{rj.job_id}")
                continue
            m, u, moved = uncertainty_guard(j, rj.matched_constraints, rj.unmet_or_uncertain)
            if moved:
                guard.append(f"moved_{moved}_unverifiable_constraint(s)_for_job:{j.id}")
            ids_kept.append(j.id)
            out.append(FinalJob(job_id=j.id, title=j.title_display, company=j.company_display, location=j.location_norm,
                                apply_urls=list(j.apply_urls or []), why=rj.why, matched_constraints=m, unmet_or_uncertain=u))
        summary = final.answer_summary
    else:
        for jid in last_search_order[:5]:  # graceful degradation: plain search order, no model claims
            j = session.get(Job, jid)
            if j is not None:
                out.append(FinalJob(job_id=j.id, title=j.title_display, company=j.company_display, location=j.location_norm,
                                    apply_urls=list(j.apply_urls or []), why="(unranked: agent could not finish)",
                                    matched_constraints=[], unmet_or_uncertain=["not verified against your constraints"]))
        summary = "Partial results: the assistant could not finish; these are raw search matches, not verified."
        if not out:
            status = "failed"
    return AgentResponse(status=status, answer_summary=summary, results=out, steps=steps,  # type: ignore[arg-type]
                         guard_events=guard, trace=trace, latency_ms=int((time.perf_counter() - t0) * 1000))
