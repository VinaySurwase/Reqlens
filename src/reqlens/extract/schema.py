from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Seniority = Literal["intern", "junior", "mid", "senior", "staff_plus", "unknown"]
RemoteType = Literal["onsite", "hybrid", "remote", "unknown"]
EmploymentType = Literal["full_time", "part_time", "contract", "internship", "unknown"]

MAX_SKILLS = 25
_ALIASES = {
    "js": "javascript", "ts": "typescript", "postgres": "postgresql", "psql": "postgresql",
    "golang": "go", "k8s": "kubernetes", "reactjs": "react", "react.js": "react", "node": "node.js",
    "nodejs": "node.js", "py": "python", "tf": "tensorflow", "sklearn": "scikit-learn",
    "gcp": "google cloud", "amazon web services": "aws", "c plus plus": "c++", "mongo": "mongodb",
}


def normalize_skill(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower()).strip(" .,;:-")
    return _ALIASES.get(s, s)


class ExtractionResult(BaseModel):
    """Validated LLM output. Nothing reaches the jobs table unless it parses into this model."""

    title_clean: str = Field(min_length=1, max_length=200)
    seniority: Seniority = "unknown"
    min_years_experience: int | None = Field(default=None, ge=0, le=30)
    max_years_experience: int | None = Field(default=None, ge=0, le=30)
    remote_type: RemoteType = "unknown"
    location_normalized: str = Field(default="", max_length=200)
    skills: list[str] = Field(default_factory=list)
    employment_type: EmploymentType = "unknown"
    is_internship: bool = False

    @field_validator("title_clean", "location_normalized")
    @classmethod
    def _strip(cls, v: str) -> str:
        return re.sub(r"\s+", " ", v).strip()

    @field_validator("skills")
    @classmethod
    def _skills(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in v:
            s = normalize_skill(str(raw))
            if not s or len(s) > 40 or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out[:MAX_SKILLS]

    @model_validator(mode="after")
    def _years_order(self) -> ExtractionResult:
        lo, hi = self.min_years_experience, self.max_years_experience
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("min_years_experience must be <= max_years_experience")
        return self


class ExtractInput(BaseModel):
    title: str
    company: str
    location: str
    description: str
