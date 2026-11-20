"""All tunables live here. Env vars use the RL_ prefix."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://ghost:ghost@localhost:5432/reqlens"
    gemini_api_key: str = ""

    # Provider switches. 'hash' / 'heuristic' exist so the whole pipeline can run offline.
    embedding_provider: str = "gemini"  # gemini | hash
    extraction_provider: str = "gemini"  # gemini | heuristic

    # Model names are NOT verified from this repo's build environment (no network there).
    # Check https://ai.google.dev/gemini-api/docs/models before relying on them.
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768  # must match the vector(768) column created by migration 0001

    # Ingestion
    expire_after_misses: int = 2
    user_agent: str = "reqlens/0.1 (portfolio project; contact: set RL_USER_AGENT)"
    http_timeout_s: float = 30.0
    http_retry_max_wait_s: float = 20.0  # 0 disables waiting (tests)
    http_cache_dir: str = ".cache/http"
    http_cache_ttl_s: int = 3600
    boards_file: str = "config/boards.yaml"

    # Extraction
    max_description_chars: int = 6000
    extraction_batch_size: int = 20
    llm_max_backoff_s: float = 60.0
    llm_max_retries_on_quota: int = 5

    # Dedup (weights/threshold are UNTUNED defaults until eval/run_dedup_eval.py is run on gold pairs)
    dedup_threshold: float = 0.70
    dedup_w_title: float = 0.45
    dedup_w_desc: float = 0.25
    dedup_w_loc: float = 0.20
    dedup_w_seniority: float = 0.10

    # Search
    rrf_k: int = 60
    candidate_depth: int = 100
    rerank_depth: int = 30
    hnsw_ef_search: int = 100
    freshness_boost: float = 0.0  # off by default; claim nothing until ablated

    # Agent
    max_tool_steps: int = 4
    agent_timeout_s: float = 45.0
    agent_max_tokens: int = 40000

    # API
    max_query_chars: int = 500
    rate_limit: str = "30/minute"


@lru_cache
def get_settings() -> Settings:
    return Settings()
