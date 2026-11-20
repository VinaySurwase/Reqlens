from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from reqlens.db.models import IngestRun, RawPosting
from reqlens.ingest.runner import run_ingest
from reqlens.sources.fixture import FixtureServer

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def snapshot(session):
    session.expire_all()
    rows = session.scalars(select(RawPosting).order_by(RawPosting.id)).all()
    return [(r.id, r.content_hash, r.first_seen_at, r.last_changed_at, r.status, r.missed_runs,
             r.extraction_status) for r in rows]


async def test_first_run_stores_everything(session):
    srv = FixtureServer()
    r = await run_ingest(session, srv.adapters()["lever"], T0)
    assert r.status == "success" and r.new == r.fetched > 0 and r.changed == r.unchanged == r.expired == 0
    assert session.scalar(select(func.count()).select_from(RawPosting).where(RawPosting.extraction_status == "pending")) == r.new


async def test_idempotent_second_run_changes_nothing(session):
    srv = FixtureServer()
    ad = srv.adapters()
    await run_ingest(session, ad["greenhouse"], T0)
    before = snapshot(session)
    r2 = await run_ingest(session, srv.adapters()["greenhouse"], T0 + timedelta(hours=12))
    assert (r2.new, r2.changed, r2.expired) == (0, 0, 0) and r2.unchanged == r2.fetched
    assert snapshot(session) == before  # identical except last_seen_at, which is not in the snapshot
    run = session.scalars(select(IngestRun).order_by(IngestRun.id.desc())).first()
    assert run.unchanged_count == r2.fetched


async def test_changed_posting_is_marked_pending_again(session):
    srv = FixtureServer()
    await run_ingest(session, srv.adapters()["lever"], T0)
    row = session.scalars(select(RawPosting)).first()
    row.extraction_status = "done"
    row.extracted = {"x": 1}
    session.commit()
    token = next(iter(srv.data["lever"]))
    srv.data["lever"][token][0]["description"] = "<p>Totally new description</p>"
    srv.data["lever"][token][0]["text"] = "Brand New Title"
    r = await run_ingest(session, srv.adapters()["lever"], T0 + timedelta(hours=12))
    assert r.changed == 1
    session.expire_all()
    changed = session.scalars(select(RawPosting).where(RawPosting.title == "Brand New Title")).one()
    assert changed.extraction_status == "pending" and changed.extracted is None and changed.last_changed_at is not None


async def test_expiry_after_two_successful_misses(session):
    srv = FixtureServer()
    await run_ingest(session, srv.adapters()["lever"], T0)
    token = next(iter(srv.data["lever"]))
    gone = srv.data["lever"][token].pop(0)
    gone_id = f"{token}:{gone['id']}"
    r1 = await run_ingest(session, srv.adapters()["lever"], T0 + timedelta(hours=12))
    row = session.scalars(select(RawPosting).where(RawPosting.source_job_id == gone_id)).one()
    assert r1.expired == 0 and row.missed_runs == 1 and row.status == "active"
    r2 = await run_ingest(session, srv.adapters()["lever"], T0 + timedelta(hours=24))
    session.expire_all()
    assert r2.expired == 1 and session.get(RawPosting, row.id).status == "expired"


async def test_never_expire_after_failed_or_partial_run(session):
    srv = FixtureServer()
    await run_ingest(session, srv.adapters()["greenhouse"], T0)
    n = session.scalar(select(func.count()).select_from(RawPosting))
    srv.fail.add("greenhouse.io")  # every board 500s -> adapter returns nothing + errors
    for i in range(1, 5):
        r = await run_ingest(session, srv.adapters()["greenhouse"], T0 + timedelta(hours=12 * i))
        assert r.status == "partial" and r.expired == 0
    session.expire_all()
    assert session.scalar(select(func.count()).select_from(RawPosting).where(RawPosting.status == "active")) == n
    assert session.scalar(select(func.max(RawPosting.missed_runs))) == 0


async def test_one_board_down_is_partial_and_does_not_expire_its_rows(session):
    srv = FixtureServer()
    await run_ingest(session, srv.adapters()["greenhouse"], T0)
    srv.fail.add("quarrylabs")
    for i in range(1, 4):
        r = await run_ingest(session, srv.adapters()["greenhouse"], T0 + timedelta(hours=12 * i))
        assert r.status == "partial" and r.expired == 0


async def test_empty_success_with_existing_postings_is_demoted(session):
    srv = FixtureServer()
    await run_ingest(session, srv.adapters()["remotive"], T0)
    srv.data["remotive"] = {"jobs": []}
    for i in range(1, 4):
        r = await run_ingest(session, srv.adapters()["remotive"], T0 + timedelta(hours=12 * i))
        assert r.status == "partial" and r.expired == 0


async def test_reappearing_posting_is_reactivated(session):
    srv = FixtureServer()
    await run_ingest(session, srv.adapters()["lever"], T0)
    token = next(iter(srv.data["lever"]))
    gone = srv.data["lever"][token].pop(0)
    for i in (1, 2):
        await run_ingest(session, srv.adapters()["lever"], T0 + timedelta(hours=12 * i))
    srv.data["lever"][token].insert(0, gone)
    await run_ingest(session, srv.adapters()["lever"], T0 + timedelta(hours=48))
    session.expire_all()
    row = session.scalars(select(RawPosting).where(RawPosting.source_job_id == f"{token}:{gone['id']}")).one()
    assert row.status == "active" and row.missed_runs == 0
