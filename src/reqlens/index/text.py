from __future__ import annotations

from typing import Any

from reqlens.normalize import text_hash

DESC_CHARS = 1500


def build_embedding_text(
    title: str, company: str, location: str, remote_type: str, seniority: str, skills: list[str],
    description: str,
) -> str:
    bits = [f"{title} at {company}"]
    if location:
        bits.append(f"Location: {location}")
    if remote_type != "unknown":
        bits.append(f"Work type: {remote_type}")
    if seniority != "unknown":
        bits.append(f"Level: {seniority}")
    if skills:
        bits.append("Skills: " + ", ".join(skills))
    bits.append(description[:DESC_CHARS])
    return "\n".join(bits)


def embedding_text_for_job(job: Any) -> str:
    return build_embedding_text(job.title_display, job.company_display, job.location_norm, job.remote_type,
                                job.seniority, list(job.skills or []), job.description_clean)


def embedding_hash(text: str) -> str:
    return text_hash(text)
