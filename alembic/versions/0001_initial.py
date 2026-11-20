"""initial schema

Revision ID: 0001
Revises:
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE sources (
        id serial PRIMARY KEY, name varchar(64) UNIQUE NOT NULL,
        last_run_at timestamptz, last_success_at timestamptz)""",
    """CREATE TABLE ingest_runs (
        id serial PRIMARY KEY, source_id int NOT NULL REFERENCES sources(id),
        started_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz,
        status varchar(16) NOT NULL DEFAULT 'failed',
        fetched_count int NOT NULL DEFAULT 0, new_count int NOT NULL DEFAULT 0,
        changed_count int NOT NULL DEFAULT 0, unchanged_count int NOT NULL DEFAULT 0,
        expired_count int NOT NULL DEFAULT 0, error_summary text)""",
    """CREATE TABLE jobs (
        id serial PRIMARY KEY, company_norm varchar(255) NOT NULL, company_display text NOT NULL DEFAULT '',
        title_norm text NOT NULL DEFAULT '', title_display text NOT NULL DEFAULT '',
        location_norm text NOT NULL DEFAULT '', remote_type varchar(16) NOT NULL DEFAULT 'unknown',
        seniority varchar(16) NOT NULL DEFAULT 'unknown',
        min_years_experience int, max_years_experience int,
        skills text[] NOT NULL DEFAULT '{}', employment_type varchar(24) NOT NULL DEFAULT 'unknown',
        is_internship boolean NOT NULL DEFAULT false, description_clean text NOT NULL DEFAULT '',
        apply_urls jsonb NOT NULL DEFAULT '[]', posted_at timestamptz,
        first_seen_at timestamptz NOT NULL, last_seen_at timestamptz NOT NULL,
        status varchar(16) NOT NULL DEFAULT 'active',
        embedding vector(768), embedding_text_hash varchar(64) NOT NULL DEFAULT '',
        tsv tsvector, source_count int NOT NULL DEFAULT 1)""",
    """CREATE TABLE raw_postings (
        id serial PRIMARY KEY, source_id int NOT NULL REFERENCES sources(id),
        source_job_id varchar(255) NOT NULL, url text NOT NULL DEFAULT '', title text NOT NULL DEFAULT '',
        company text NOT NULL DEFAULT '', company_norm varchar(255) NOT NULL DEFAULT '',
        location_raw text NOT NULL DEFAULT '', description_raw text NOT NULL DEFAULT '',
        description_clean text NOT NULL DEFAULT '', posted_at timestamptz,
        content_hash varchar(64) NOT NULL, raw_payload jsonb,
        first_seen_at timestamptz NOT NULL, last_seen_at timestamptz NOT NULL, last_changed_at timestamptz,
        missed_runs int NOT NULL DEFAULT 0, status varchar(16) NOT NULL DEFAULT 'active',
        job_id int REFERENCES jobs(id), extraction_status varchar(16) NOT NULL DEFAULT 'pending',
        extracted jsonb, extraction_error text, embedding vector(768),
        dedup_dirty boolean NOT NULL DEFAULT true,
        CONSTRAINT uq_raw_source_job UNIQUE (source_id, source_job_id))""",
    """CREATE TABLE llm_calls (
        id bigserial PRIMARY KEY, purpose varchar(16) NOT NULL, model varchar(64) NOT NULL,
        input_tokens int NOT NULL DEFAULT 0, output_tokens int NOT NULL DEFAULT 0,
        latency_ms int NOT NULL DEFAULT 0, cost_usd_estimate double precision NOT NULL DEFAULT 0,
        ok boolean NOT NULL DEFAULT true, created_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX ix_raw_company_status ON raw_postings (company_norm, status)",
    "CREATE INDEX ix_raw_extraction_status ON raw_postings (extraction_status)",
    "CREATE INDEX ix_raw_job_id ON raw_postings (job_id)",
    "CREATE INDEX ix_jobs_company_status ON jobs (company_norm, status)",
    "CREATE INDEX ix_jobs_tsv ON jobs USING gin (tsv)",
    # HNSW: no training step, supports incremental inserts (jobs are added/updated continuously).
    "CREATE INDEX ix_jobs_embedding ON jobs USING hnsw (embedding vector_cosine_ops)",
]


def upgrade() -> None:
    for stmt in DDL:
        op.execute(stmt)


def downgrade() -> None:
    for t in ["raw_postings", "llm_calls", "jobs", "ingest_runs", "sources"]:
        op.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
