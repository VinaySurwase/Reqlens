"""Generate a SYNTHETIC multi-source job corpus (fictional companies) with ground-truth duplicate groups.

Output is API-shaped JSON (shapes are from memory of the real APIs -- unverified) under tests/fixtures/.
Used for adapter/pipeline tests and the offline SMOKE eval. NOT real data: never quote numbers from it
as results.  Usage: python scripts/make_fixtures.py [--seed 7]
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

COMPANIES = [
    ("Northwind Robotics", "northwindrobotics", "greenhouse", "Northwind Robotics, Inc."),
    ("Lumen Analytics", "lumenanalytics", "lever", "Lumen Analytics Inc"),
    ("Quarry Labs", "quarrylabs", "greenhouse", "Quarry Labs"),
    ("Helio Health", "heliohealth", "lever", "Helio Health Technologies"),
    ("Brightwave", "brightwave", "greenhouse", "Brightwave Ltd."),
    ("Orbit Freight", "orbitfreight", "lever", "Orbit Freight"),
    ("Kestrel AI", "kestrelai", "greenhouse", "Kestrel AI, Inc."),
    ("Mosaic Payments", "mosaicpayments", "lever", "Mosaic Payments LLC"),
    ("Tundra Cloud", "tundracloud", "greenhouse", "Tundra Cloud"),
    ("Verdant Energy", "verdantenergy", "lever", "Verdant Energy Corp"),
    ("Pillar Games", "pillargames", "greenhouse", "Pillar Games"),
    ("Sable Security", "sablesecurity", "lever", "Sable Security, Inc."),
]
CITIES = ["San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "London, UK", "Bengaluru, India", "Berlin, Germany"]
ROLES = {
    "Backend Engineer": ["python", "postgresql", "fastapi", "docker", "aws", "redis"],
    "Machine Learning Engineer": ["python", "pytorch", "machine learning", "docker", "kubernetes", "mlops"],
    "Data Engineer": ["python", "sql", "spark", "airflow", "snowflake", "kafka"],
    "Frontend Engineer": ["typescript", "react", "javascript", "graphql", "git"],
    "Site Reliability Engineer": ["kubernetes", "terraform", "linux", "aws", "go", "docker"],
    "Mobile Engineer": ["kotlin", "swift", "android", "ios", "git"],
    "Full Stack Engineer": ["javascript", "node.js", "react", "postgresql", "docker"],
    "Data Scientist": ["python", "pandas", "sql", "scikit-learn", "machine learning"],
    "Security Engineer": ["python", "linux", "aws", "go", "terraform"],
    "NLP Research Engineer": ["python", "pytorch", "nlp", "llm", "rag"],
}
LEVELS = {  # level -> (title prefix, min_years, max_years)
    "junior": ("Junior ", 0, 2), "mid": ("", 2, 4), "senior": ("Senior ", 5, 8), "staff_plus": ("Staff ", 8, None),
    "intern": ("", None, None),
}
TEAMS = ["Platform", "Growth", "Payments", "Core", "Infrastructure"]


def build_jobs(rng: random.Random) -> list[dict]:
    jobs: list[dict] = []
    uid = 0

    def add(comp, role, level, loc, remote, skills, team=None):
        nonlocal uid
        uid += 1
        prefix, lo, hi = LEVELS[level]
        title = (f"{role} Intern" if level == "intern" else f"{prefix}{role}")
        jobs.append(dict(uid=uid, company=comp, role=role, level=level, title=title, loc=loc, remote=remote,
                         skills=skills, team=team, min_years=lo, max_years=hi))

    for comp_t in COMPANIES:
        comp = comp_t[0]
        roles = rng.sample(list(ROLES), 6)
        for role in roles:
            level = rng.choice(["junior", "mid", "mid", "senior", "senior", "staff_plus"])
            remote = rng.choice(["remote", "hybrid", "onsite", "onsite"])
            loc = "Remote" if remote == "remote" else rng.choice(CITIES)
            add(comp, role, level, loc, remote, ROLES[role])
        # hard negative 1: same title, different city (not duplicates)
        role = roles[0]
        base = jobs[-6]
        other = rng.choice([c for c in CITIES if c != base["loc"]])
        add(comp, base["role"], base["level"], other, "onsite", ROLES[base["role"]])
        # hard negative 2: internship version of a role
        add(comp, roles[1], "intern", rng.choice(CITIES), "onsite", ROLES[roles[1]])
        # hard negative 3: same title, different team (different skill emphasis)
        b2 = jobs[-8]
        add(comp, b2["role"], b2["level"], b2["loc"], b2["remote"], rng.sample(ROLES[b2["role"]], 3) + ["graphql"],
            team=rng.choice(TEAMS))
    return jobs


def years_line(j: dict) -> str:
    if j["level"] == "intern":
        return "No prior industry experience required; currently enrolled in a CS degree."
    if j["max_years"]:
        return f"{j['min_years']}-{j['max_years']} years of professional software experience."
    return f"{j['min_years']}+ years of professional software experience."


def core(j: dict) -> str:
    team = f"You will join the {j['team']} team. " if j["team"] else ""
    return (f"{team}You will design, build and operate services as a {j['role'].lower()}. "
            f"Requirements: {years_line(j)} Hands-on experience with {', '.join(j['skills'])}. "
            "Strong communication and a bias for shipping.")


def where(j: dict) -> str:
    return "Remote" if j["remote"] == "remote" else f"{j['loc']} ({'hybrid' if j['remote']=='hybrid' else 'on-site'})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="tests/fixtures")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    jobs = build_jobs(rng)
    comp_meta = {c[0]: c for c in COMPANIES}
    gh: dict[str, dict] = {}
    lv: dict[str, list] = {}
    hn_children: list[dict] = []
    remotive: list[dict] = []
    truth: list[dict] = []
    counter = 1000

    for j in jobs:
        name, token, ats, legal = comp_meta[j["company"]]
        counter += 1
        if ats == "greenhouse":
            gh.setdefault(token, {"jobs": []})["jobs"].append({
                "id": counter, "title": j["title"], "absolute_url": f"https://boards.example.test/{token}/jobs/{counter}",
                "location": {"name": j["loc"] if j["remote"] != "remote" else "Remote"},
                "content": html.escape(f"<p>{core(j)}</p><ul><li>{where(j)}</li></ul>"),
                "updated_at": "2026-10-01T10:00:00Z", "company_name": name})
            truth.append(dict(source="greenhouse", source_job_id=f"{token}:{counter}", uid=j["uid"]))
        else:
            lv.setdefault(token, []).append({
                "id": f"lv-{counter}", "text": j["title"], "hostedUrl": f"https://jobs.example.test/{token}/lv-{counter}",
                "categories": {"location": j["loc"] if j["remote"] != "remote" else "Remote", "team": j["team"] or "Engineering",
                               "commitment": "Intern" if j["level"] == "intern" else "Full-time"},
                "workplaceType": j["remote"] if j["remote"] != "onsite" else "on-site",
                "description": f"<div>{core(j)} Location: {where(j)}.</div>", "createdAt": 1790000000000})
            truth.append(dict(source="lever", source_job_id=f"{token}:lv-{counter}", uid=j["uid"]))
        if rng.random() < 0.35:  # also on HN, rendered differently, company name variant
            counter += 1
            line = f"{legal} | {j['title']} | {j['loc'] if j['remote']!='remote' else 'Remote'} | {'REMOTE' if j['remote']=='remote' else 'ONSITE'}"
            hn_children.append({"id": counter, "author": "x", "created_at_i": 1790000500,
                                "text": f"{html.escape(line)}<p>We are hiring a {html.escape(j['title'])}. {html.escape(core(j))} Email jobs@example.test</p>"})
            truth.append(dict(source="hn", source_job_id=str(counter), uid=j["uid"]))
        if j["remote"] == "remote" and rng.random() < 0.5:
            counter += 1
            remotive.append({"id": counter, "url": f"https://remotive.example.test/{counter}", "title": j["title"],
                             "company_name": legal, "candidate_required_location": "Worldwide",
                             "description": f"<p>{core(j)}</p>", "publication_date": "2026-10-02T09:00:00"})
            truth.append(dict(source="remotive", source_job_id=str(counter), uid=j["uid"]))
    for k in range(6):  # messy HN comments with no parseable company line
        counter += 1
        hn_children.append({"id": counter, "author": "y", "created_at_i": 1790000600,
                            "text": f"hiring engineers, remote ok. We build things with python and go. reach out {k}"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "greenhouse.json").write_text(json.dumps(gh, indent=1))
    (out / "lever.json").write_text(json.dumps(lv, indent=1))
    (out / "hn.json").write_text(json.dumps({"id": 9000001, "children": hn_children}, indent=1))
    (out / "remotive.json").write_text(json.dumps({"jobs": remotive}, indent=1))
    (out / "truth.json").write_text(json.dumps({"jobs": jobs, "postings": truth,
                                                "boards": {"greenhouse": [{"token": c[1], "company": c[0]} for c in COMPANIES if c[2] == "greenhouse"],
                                                           "lever": [{"token": c[1], "company": c[0]} for c in COMPANIES if c[2] == "lever"]}}, indent=1))
    print(f"jobs={len(jobs)} postings={len(truth)} -> {out}")


if __name__ == "__main__":
    main()
