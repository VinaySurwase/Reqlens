import pytest

from reqlens.sources.fixture import FixtureServer
from reqlens.sources.hn import parse_first_line


async def collect(adapter):
    return [p async for p in adapter.fetch()]


async def test_all_adapters_parse_fixtures():
    srv = FixtureServer()
    for name, ad in srv.adapters().items():
        posts = await collect(ad)
        assert posts, name
        assert not ad.errors, (name, ad.errors)
        assert all(p.source == name and p.source_job_id and p.title is not None for p in posts)
    gh = await collect(srv.adapters()["greenhouse"])
    assert all(p.url.startswith("https://") and p.company for p in gh)


async def test_board_failure_is_isolated_and_recorded():
    srv = FixtureServer()
    srv.fail.add("quarrylabs")
    ad = srv.adapters()["greenhouse"]
    posts = await collect(ad)
    assert posts  # other boards still returned
    assert any("quarrylabs" in e for e in ad.errors)


async def test_hn_missing_thread_records_error():
    srv = FixtureServer()
    srv.fail.add("search_by_date")
    ad = srv.adapters()["hn"]
    assert await collect(ad) == []
    assert ad.errors


@pytest.mark.parametrize("line,company", [
    ("Acme | Senior Backend Engineer | Remote | Full-time", "Acme"),
    ("no pipes here at all", ""),
])
def test_hn_first_line(line, company):
    assert parse_first_line(line)[0] == company
