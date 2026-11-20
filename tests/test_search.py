import pytest
from pydantic import ValidationError
from sqlalchemy import select

from reqlens.db.models import Job
from reqlens.extract.heuristic import HeuristicExtractor
from reqlens.index.embedder import HashEmbedder
from reqlens.llm import QuotaError, Usage
from reqlens.pipeline import run_pipeline
from reqlens.search.engine import lexical_tsquery, rrf, search
from reqlens.search.filters import SearchFilters
from reqlens.sources.fixture import FixtureServer


def test_rrf_math_and_ties():
    out = dict(rrf([[1, 2, 3], [3, 2, 4]], k=60))
    assert out[2] == pytest.approx(1 / 62 + 1 / 62)
    assert out[1] == pytest.approx(1 / 61) and out[4] == pytest.approx(1 / 63)
    assert rrf([[1, 2], [2, 1]], k=60)[0][0] == 1  # exact tie -> lower id first (deterministic)
    assert rrf([]) == [] and rrf([[], []]) == []


def test_tsquery_is_or_and_safe():
    q = lexical_tsquery("I want a senior Python backend job, no more than 5 years!!'; DROP TABLE x;--")
    assert "python" in q and "backend" in q and "'" not in q and ";" not in q and " | " in q
    assert lexical_tsquery("the and of") == ""


def test_filters_reject_unknown_keys_and_bad_values():
    with pytest.raises(ValidationError):
        SearchFilters(salary_over=100)
    with pytest.raises(ValidationError):
        SearchFilters(max_min_years_experience=99)
    with pytest.raises(ValidationError):
        SearchFilters(remote_type=["moon"])


@pytest.fixture
async def corpus(session):
    srv = FixtureServer()
    await run_pipeline(session, list(srv.adapters().values()), HeuristicExtractor(), HashEmbedder())
    return session


E = HashEmbedder()


async def test_every_mode_returns_ranked_results(corpus):
    for mode in ("lexical", "dense", "hybrid"):
        r = search(corpus, "backend engineer python postgresql", mode, 10, embedder=E)
        assert r.results and r.mode_used == mode and r.degraded is None
        assert len(r.results) <= 10 and all(h.apply_urls for h in r.results)
    r = search(corpus, "backend engineer python", "lexical", 5)
    assert "backend" in r.results[0].title.lower() or "python" in " ".join(r.results[0].skills)


async def test_filters_are_enforced_in_all_modes(corpus):
    f = SearchFilters(remote_type=["remote"], seniority=["senior"])
    for mode in ("lexical", "dense", "hybrid"):
        r = search(corpus, "engineer", mode, 20, f, embedder=E)
        assert r.results, mode
        assert all(h.remote_type == "remote" and h.seniority == "senior" for h in r.results)
    f2 = SearchFilters(max_min_years_experience=2, include_unknown_years=False)
    r = search(corpus, "engineer", "hybrid", 30, f2, embedder=E)
    assert r.results and all(h.min_years_experience is not None and h.min_years_experience <= 2 for h in r.results)
    f3 = SearchFilters(skills_all=["python", "postgresql"], exclude_seniority=["senior", "staff_plus"])
    for h in search(corpus, "engineer", "hybrid", 30, f3, embedder=E).results:
        assert {"python", "postgresql"} <= set(h.skills) and h.seniority not in ("senior", "staff_plus")


async def test_expired_jobs_never_returned_by_default(corpus):
    jid = corpus.scalars(select(Job.id)).first()
    corpus.query(Job).filter_by(id=jid).update({"status": "expired"})
    corpus.commit()
    for mode in ("lexical", "dense", "hybrid"):
        assert jid not in [h.job_id for h in search(corpus, "engineer", mode, 100, embedder=E).results]


class BrokenEmbedder:
    model, dim = "x", 768

    def embed(self, texts, kind="doc"):
        raise QuotaError("429")


async def test_embedding_outage_degrades_to_lexical(corpus):
    r = search(corpus, "backend engineer python", "hybrid", 5, embedder=BrokenEmbedder())
    assert r.mode_used == "lexical" and "fell back" in r.degraded and r.results


class RerankLLM:
    model = "fake"

    def __init__(self, fn):
        self.fn = fn

    def generate_structured(self, *, system, prompt, schema, temperature=0.0):
        return self.fn(prompt, schema), Usage(1, 1, 1)


async def test_rerank_reorders_and_ignores_hallucinated_ids(corpus):
    base = search(corpus, "engineer", "hybrid", 10, embedder=E)
    last = base.results[-1].job_id

    def fn(prompt, schema):
        return schema(scores=[{"job_id": last, "score": 3}, {"job_id": 999999, "score": 3}])
    r = search(corpus, "engineer", "hybrid_rerank", 10, embedder=E, llm=RerankLLM(fn))
    assert r.mode_used == "hybrid_rerank" and r.results[0].job_id == last
    assert 999999 not in [h.job_id for h in r.results]


async def test_rerank_failure_falls_back_to_hybrid_order(corpus):
    base = search(corpus, "engineer", "hybrid", 10, embedder=E)

    def bad(prompt, schema):
        return schema.model_validate({"scores": [{"job_id": 1, "score": 9}]})  # invalid -> ValidationError
    r = search(corpus, "engineer", "hybrid_rerank", 10, embedder=E, llm=RerankLLM(bad))
    assert r.mode_used == "hybrid" and r.degraded and [h.job_id for h in r.results] == [h.job_id for h in base.results]
    r2 = search(corpus, "engineer", "hybrid_rerank", 10, embedder=E, llm=None)
    assert r2.mode_used == "hybrid" and r2.degraded
