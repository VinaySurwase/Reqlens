"""Thin LLM / embedding abstraction. Production = Gemini (google-genai); tests/offline = fakes.

NOTE: not exercised against the live Gemini API from the build environment (no network egress there).
SDK type/field names were checked by introspecting google-genai 3.x; model names and quotas must be
verified by the owner (docs/SOURCES.md, "Gemini" section).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from reqlens.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class QuotaError(Exception):
    """Rate limit / quota exhausted (HTTP 429). Callers back off and resume later."""


class TransientLLMError(Exception):
    """5xx / network problems. Safe to retry."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class LLMClient(Protocol):
    model: str

    def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], temperature: float = 0.0
    ) -> tuple[T, Usage]:
        """Return a *validated* instance of ``schema`` or raise pydantic.ValidationError."""
        ...


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], kind: str = "doc") -> list[list[float]]: ...


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _map_error(e: Exception) -> Exception:
    code = getattr(e, "code", None)
    if code == 429:
        return QuotaError(str(e))
    if isinstance(code, int) and code >= 500:
        return TransientLLMError(str(e))
    return e


def _tok(meta: Any, *names: str) -> int:
    for n in names:
        v = getattr(meta, n, None)
        if isinstance(v, int):
            return v
    return 0


class GeminiClient:
    def __init__(self, settings: Settings | None = None) -> None:
        from google import genai  # imported lazily so offline tests never need the key

        s = settings or get_settings()
        if not s.gemini_api_key:
            raise RuntimeError("RL_GEMINI_API_KEY is not set")
        self._genai = genai
        self._client = genai.Client(api_key=s.gemini_api_key)
        self.model = s.gemini_llm_model

    def generate_structured(
        self, *, system: str, prompt: str, schema: type[T], temperature: float = 0.0
    ) -> tuple[T, Usage]:
        from google.genai import types

        t0 = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )
        except Exception as e:  # noqa: BLE001
            raise _map_error(e) from e
        usage = Usage(
            _tok(resp.usage_metadata, "prompt_token_count"),
            _tok(resp.usage_metadata, "candidates_token_count", "response_token_count"),
            int((time.perf_counter() - t0) * 1000),
        )
        return schema.model_validate_json(resp.text or ""), usage

    # --- function calling (agent) -------------------------------------------------------------
    def raw_client(self) -> Any:
        return self._client


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        from google import genai

        s = settings or get_settings()
        if not s.gemini_api_key:
            raise RuntimeError("RL_GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=s.gemini_api_key)
        self.model = s.gemini_embed_model
        self.dim = s.embed_dim

    def embed(self, texts: list[str], kind: str = "doc") -> list[list[float]]:
        from google.genai import types

        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # batch; 100 is a conservative per-request size
            try:
                resp = self._client.models.embed_content(
                    model=self.model,
                    contents=texts[i : i + 100],
                    config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
                )
            except Exception as e:  # noqa: BLE001
                raise _map_error(e) from e
            embs = resp.embeddings or []
            if len(embs) != len(texts[i : i + 100]):
                raise TransientLLMError("embedding count mismatch")
            out.extend(_l2(list(e.values or [])) for e in embs)
        return out


def build_embedder(settings: Settings | None = None) -> Embedder:
    from reqlens.index.embedder import HashEmbedder

    s = settings or get_settings()
    if s.embedding_provider == "hash":
        return HashEmbedder(s.embed_dim)
    return GeminiEmbedder(s)
