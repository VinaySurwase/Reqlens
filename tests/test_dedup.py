import itertools
from collections import defaultdict

import numpy as np
from sqlalchemy import select

from reqlens.db.models import Job, RawPosting, Source
from reqlens.dedup.score import DedupItem, Weights, cluster, score_pair
from reqlens.extract.heuristic import HeuristicExtractor
from reqlens.index.embedder import HashEmbedder
from reqlens.pipeline import run_pipeline
from reqlens.sources.fixture import FixtureServer

W = Weights()


def item(i, title, loc="San Francisco, CA", remote="onsite", sen="mid", intern=False, desc="build services python postgres", skills=None):
    return DedupItem(i, title, loc, desc, {"seniority": sen, "remote_type": remote, "is_internship": intern,
                                           "location_normalized": loc}, None)


def test_true_duplicate_scores_high():
    r = score_pair(item(1, "Backend Engineer"), item(2, "Backend Developer"), W)
    assert r.veto is None and r.score > 0.85


def test_hard_negative_different_city_vetoed():
    r = score_pair(item(1, "Backend Engineer", loc="San Francisco, CA"), item(2, "Backend Engineer", loc="Berlin, Germany"), W)
    assert r.veto == "location_conflict" and r.score == 0


def test_hard_negative_internship_vetoed():
    r = score_pair(item(1, "Software Engineer"), item(2, "Software Engineer Intern", intern=True), W)
    assert r.veto == "internship_mismatch"


def test_hard_negative_junior_vs_senior_same_title():
    r = score_pair(item(1, "Software Engineer", sen="junior"), item(2, "Software Engineer", sen="senior"), W)
    assert r.veto == "seniority_distance"


def test_remote_and_sf_not_vetoed_and_unknown_location_neutral():
    a, b = item(1, "ML Engineer", loc="Remote", remote="remote"), item(2, "ML Engineer", loc="San Francisco, CA")
    assert score_pair(a, b, W).veto is None
    c = item(3, "ML Engineer", loc="")
    assert score_pair(a, c, W).veto is None


def test_cluster_does_not_chain_across_vetoes():
    # A~B (same city) and B~C (remote), but A and C are in conflicting cities -> must not become one cluster
    a = item(1, "Data Engineer", loc="Berlin, Germany")
    b = item(2, "Data Engineer", loc="Remote", remote="remote")
    c = item(3, "Data Engineer", loc="Austin, TX")
    groups = cluster([a, b, c], threshold=0.6, w=W)
    assert not any({1, 3} <= set(g) for g in groups)
    assert sorted(itertools.chain(*groups)) == [1, 2, 3]


def test_cluster_uses_embeddings_when_present():
    v = np.array([1.0, 0, 0])
    a, b = item(1, "Engineer"), item(2, "Engineer")
    a.embedding, b.embedding = v, np.array([0, 1.0, 0])  # orthogonal descriptions
    assert score_pair(a, b, W).parts["desc"] == 0.0


async def full_run(session, threshold=None):
    srv = FixtureServer()
    await run_pipeline(session, list(srv.adapters().values()), HeuristicExtractor(), HashEmbedder(), threshold)
    return srv


def pair_metrics(session, srv):
    names = {s.id: s.name for s in session.scalars(select(Source))}
    uid = {(p["source"], p["source_job_id"]): p["uid"] for p in srv.truth["postings"]}
    rows = session.scalars(select(RawPosting).where(RawPosting.job_id.is_not(None))).all()
    by_truth, by_pred = defaultdict(set), defaultdict(set)
    for r in rows:
        u = uid.get((names[r.source_id], r.source_job_id))
        if u is not None:
            by_truth[u].add(r.id)
            by_pred[r.job_id].add(r.id)

    def pairs(groups):
        return {frozenset(p) for g in groups.values() for p in itertools.combinations(sorted(g), 2)}
    t, p = pairs(by_truth), pairs(by_pred)
    tp = len(t & p)
    return tp / len(p) if p else 1.0, tp / len(t) if t else 1.0


async def test_pipeline_on_synthetic_corpus_dedups_without_merging_hard_negatives(session):
    srv = await full_run(session, 0.70)
    prec, rec = pair_metrics(session, srv)
    print(f"\nSYNTHETIC pair precision={prec:.3f} recall={rec:.3f}")
    assert prec >= 0.95  # hard negatives must not be merged
    assert rec >= 0.80
    jobs = session.scalars(select(Job)).all()
    raws = session.scalars(select(RawPosting).where(RawPosting.extraction_status == "done")).all()
    assert len(jobs) < len(raws)
    assert any(j.source_count > 1 for j in jobs)
    assert all(j.title_display and j.apply_urls for j in jobs)
    assert {"greenhouse", "lever", "hn", "remotive"} >= {u["source"] for j in jobs for u in j.apply_urls}


async def test_dedup_is_idempotent_and_incremental(session):
    await full_run(session)
    ids = sorted(j.id for j in session.scalars(select(Job)))
    from reqlens.dedup.merge import dedup_dirty
    st = dedup_dirty(session, HashEmbedder())
    assert st.companies == 0 and st.jobs_created == 0
    assert sorted(j.id for j in session.scalars(select(Job))) == ids


async def test_expired_postings_expire_job_when_all_sources_gone(session):
    from datetime import UTC, datetime, timedelta

    from reqlens.dedup.merge import dedup_dirty, refresh_job_liveness
    from reqlens.ingest.runner import run_ingest
    srv = await full_run(session)
    # find a job that lives only on one lever board and remove it from the feed twice
    t0 = datetime.now(UTC)
    token = next(iter(srv.data["lever"]))
    gone = srv.data["lever"][token].pop(0)
    sjid = f"{token}:{gone['id']}"
    for i in (1, 2):
        await run_ingest(session, srv.adapters()["lever"], t0 + timedelta(hours=12 * i))
    dedup_dirty(session, HashEmbedder())
    refresh_job_liveness(session)
    session.expire_all()
    raw = session.scalars(select(RawPosting).where(RawPosting.source_job_id == sjid)).one()
    job = session.get(Job, raw.job_id)
    siblings = session.scalars(select(RawPosting).where(RawPosting.job_id == job.id, RawPosting.status == "active")).all()
    assert raw.status == "expired"
    assert (job.status == "active") == bool(siblings)
