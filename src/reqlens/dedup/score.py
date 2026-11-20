"""Pure, explainable pair scoring + clustering for cross-source dedup (spec 6.3). No DB, no I/O.

score = w_title*title_sim + w_desc*desc_sim + w_loc*loc_compat + w_sen*seniority_agree
Hard VETOES (score forced to 0, and a cluster merge is refused if ANY cross-member pair is vetoed):
  * two postings from the SAME source with different ids (a source's own ids are its identity for an opening)
  * internship vs non-internship (the "intern version vs full-time version" hard negative)
  * both locations known, neither remote, and no overlapping place tokens (same title, different cities)
  * seniority levels two or more steps apart (e.g. junior vs senior with the same title)
Same title on different teams has no structured field to veto on; it is handled by description
similarity and is a documented weak spot (see docs/ARCHITECTURE.md, Limitations).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reqlens.normalize import place_sets, title_tokens

SENIORITY_RANK = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "staff_plus": 4}


@dataclass
class DedupItem:
    id: int
    title: str
    location_raw: str = ""
    description: str = ""
    extracted: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    source: str = ""
    source_job_id: str = ""


@dataclass
class Weights:
    title: float = 0.45
    desc: float = 0.25
    loc: float = 0.20
    seniority: float = 0.10


@dataclass
class PairResult:
    score: float
    veto: str | None
    parts: dict[str, float]


def title_similarity(a: str, b: str) -> float:
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    return 2 * len(sa & sb) / (len(sa) + len(sb))  # Dice coefficient on normalized token sets


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#]{3,}", text.lower()[:3000]))


def desc_similarity(a: DedupItem, b: DedupItem) -> float:
    if a.embedding is not None and b.embedding is not None:
        na, nb = np.linalg.norm(a.embedding), np.linalg.norm(b.embedding)
        if na and nb:
            return float(max(0.0, np.dot(a.embedding, b.embedding) / (na * nb)))
    wa, wb = _words(a.description), _words(b.description)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _loc(item: DedupItem) -> tuple[list[set[str]], str]:
    text = item.extracted.get("location_normalized") or item.location_raw
    return place_sets(text), item.extracted.get("remote_type", "unknown")


def location_compat(a: DedupItem, b: DedupItem) -> tuple[float, bool]:
    """Return (score, veto)."""
    ta, ra = _loc(a)
    tb, rb = _loc(b)
    if ra == "remote" and rb == "remote":
        return 1.0, False
    if ta and tb:
        if any(x & y for x in ta for y in tb):  # any listed place in common
            return 1.0, False
        if "remote" in (ra, rb):
            return 0.4, False  # a remote role may also list a home-base city
        return 0.0, True
    if "remote" in (ra, rb) and (ta or tb):
        return 0.4, False
    return 0.5, False  # unknown on one side: neutral, not evidence either way


def seniority_agreement(a: DedupItem, b: DedupItem) -> tuple[float, bool]:
    sa, sb = a.extracted.get("seniority", "unknown"), b.extracted.get("seniority", "unknown")
    if sa not in SENIORITY_RANK or sb not in SENIORITY_RANK:
        return 0.5, False
    d = abs(SENIORITY_RANK[sa] - SENIORITY_RANK[sb])
    return (1.0, False) if d == 0 else (0.3, False) if d == 1 else (0.0, True)


def score_pair(a: DedupItem, b: DedupItem, w: Weights) -> PairResult:
    ia, ib = bool(a.extracted.get("is_internship")), bool(b.extracted.get("is_internship"))
    loc, loc_veto = location_compat(a, b)
    sen, sen_veto = seniority_agreement(a, b)
    parts = {
        "title": title_similarity(a.title or a.extracted.get("title_clean", ""),
                                  b.title or b.extracted.get("title_clean", "")),
        "desc": desc_similarity(a, b),
        "loc": loc,
        "seniority": sen,
    }
    total = w.title * parts["title"] + w.desc * parts["desc"] + w.loc * parts["loc"] + w.seniority * parts["seniority"]
    same_source = bool(a.source) and a.source == b.source and a.source_job_id != b.source_job_id
    veto = (
        "same_source_distinct_id" if same_source
        else "internship_mismatch" if ia != ib
        else "location_conflict" if loc_veto
        else "seniority_distance" if sen_veto
        else None
    )
    return PairResult(0.0 if veto else total, veto, parts)


def cluster(items: list[DedupItem], threshold: float, w: Weights | None = None) -> list[list[int]]:
    """Greedy agglomeration, best pairs first; refuses a merge if any cross-cluster pair is vetoed.
    Returns clusters as lists of item ids (every item appears exactly once)."""
    w = w or Weights()
    n = len(items)
    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    vetoed: set[tuple[int, int]] = set()
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = score_pair(items[i], items[j], w)
            if r.veto:
                vetoed.add((i, j))
            elif r.score >= threshold:
                pairs.append((r.score, i, j))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in sorted(pairs, key=lambda t: (-t[0], t[1], t[2])):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if any((min(a, b), max(a, b)) in vetoed for a in members[ri] for b in members[rj]):
            continue
        parent[rj] = ri
        members[ri].extend(members.pop(rj))
    return [sorted(items[k].id for k in m) for m in members.values()]
