from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

TEST_URL = os.environ.get("RL_TEST_DATABASE_URL", "postgresql+psycopg://ghost:ghost@localhost:5432/reqlens_test")
os.environ.setdefault("RL_DATABASE_URL", TEST_URL)
os.environ.setdefault("RL_EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("RL_EXTRACTION_PROVIDER", "heuristic")
os.environ["RL_HTTP_CACHE_TTL_S"] = "0"
os.environ["RL_HTTP_RETRY_MAX_WAIT_S"] = "0"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def engine():  # type: ignore[no-untyped-def]
    eng = create_engine(TEST_URL)
    with eng.begin() as c:  # fresh schema via the REAL migration (so migrations are tested)
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    env = {**os.environ, "ALEMBIC_URL": TEST_URL}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)
    return eng


@pytest.fixture
def session(engine) -> Iterator[Session]:  # type: ignore[no-untyped-def]
    with engine.begin() as c:
        c.execute(text("TRUNCATE llm_calls, raw_postings, jobs, ingest_runs, sources RESTART IDENTITY CASCADE"))
    s = sessionmaker(engine, expire_on_commit=False)()
    yield s
    s.close()


@pytest.fixture
def session_factory(engine):  # type: ignore[no-untyped-def]
    return sessionmaker(engine, expire_on_commit=False)
