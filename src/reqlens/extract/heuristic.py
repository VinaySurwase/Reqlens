"""Regex baseline extractor. Used (a) offline/dev/tests, (b) as the BASELINE row in extraction eval.
It is deliberately simple; it is not a substitute for the LLM extractor."""
from __future__ import annotations

import re

from reqlens.extract.schema import ExtractInput, ExtractionResult, normalize_skill
from reqlens.normalize import normalize_text

SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "ruby", "php", "scala", "kotlin",
    "swift", "sql", "postgresql", "mysql", "mongodb", "redis", "kafka", "spark", "airflow", "dbt", "snowflake",
    "react", "vue", "angular", "node.js", "django", "flask", "fastapi", "spring", "graphql", "rest",
    "docker", "kubernetes", "terraform", "aws", "azure", "google cloud", "linux", "git", "ci/cd",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "machine learning", "deep learning", "nlp",
    "llm", "computer vision", "rag", "mlops", "data engineering", "elasticsearch", "ios", "android",
]
_SKILL_RE = {s: re.compile(r"(?<![a-z0-9+#])" + re.escape(s) + r"(?![a-z0-9+#])", re.I) for s in SKILL_VOCAB}
_RANGE = re.compile(r"(?<!\d)(\d{1,2})\s*(?:\+)?\s*(?:-|–|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I)
_SINGLE = re.compile(r"(?<!\d)(\d{1,2})\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|professional|industry|hands-on))?"
                     r"(?:\s+(?:work\s+)?experience)?", re.I)


def extract_years(text: str) -> tuple[int | None, int | None]:
    m = _RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= hi <= 30:
            return lo, hi
    m = _SINGLE.search(text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 30:
            return v, None
    return None, None


class HeuristicExtractor:
    model = "heuristic-v1"

    def extract(self, inp: ExtractInput):  # type: ignore[no-untyped-def]
        from reqlens.extract.extractor import ExtractOutcome

        title = normalize_text(inp.title) or "Untitled"
        text = f"{title}\n{inp.location}\n{inp.description}"
        low_title = title.lower()
        lo, hi = extract_years(inp.description)

        if re.search(r"\bintern(ship)?\b", low_title):
            sen = "intern"
        elif re.search(r"\b(staff|principal|distinguished|director|head of|vp)\b", low_title):
            sen = "staff_plus"
        elif re.search(r"\b(senior|sr\.?|lead)\b", low_title):
            sen = "senior"
        elif re.search(r"\b(junior|jr\.?|entry|graduate|new grad|associate)\b", low_title):
            sen = "junior"
        elif re.search(r"\b(ii|2|mid)\b", low_title):
            sen = "mid"
        elif lo is not None:
            sen = "senior" if lo >= 5 else "mid" if lo >= 3 else "junior"
        else:
            sen = "unknown"

        head = f"{inp.location}\n{inp.description[:800]}".lower()
        if re.search(r"\bhybrid\b", head):
            remote = "hybrid"
        elif re.search(r"(?<!not )(?<!no )\bremote\b", head) and not re.search(r"\b(not|no) remote\b", head):
            remote = "remote"
        elif re.search(r"\bon-?site\b|\bin[- ]office\b", head):
            remote = "onsite"
        else:
            remote = "unknown"

        low = text.lower()
        if sen == "intern":
            emp = "internship"
        elif re.search(r"\bcontract(or)?\b|\bfreelance\b", low):
            emp = "contract"
        elif re.search(r"\bpart[- ]time\b", low):
            emp = "part_time"
        elif re.search(r"\bfull[- ]time\b", low):
            emp = "full_time"
        else:
            emp = "unknown"

        skills = [normalize_skill(s) for s, rx in _SKILL_RE.items() if rx.search(text)]
        loc = normalize_text(inp.location) or ("Remote" if remote == "remote" else "")
        res = ExtractionResult(
            title_clean=title[:200], seniority=sen, min_years_experience=lo, max_years_experience=hi,
            remote_type=remote, location_normalized=loc[:200], skills=skills, employment_type=emp,
            is_internship=sen == "intern" or emp == "internship",
        )
        return ExtractOutcome(result=res, error=None, calls=[], retried=False)
