"""Deterministic offline embedder (signed feature hashing over unigrams+bigrams).

DEV/TEST ONLY. It captures lexical overlap, not semantics, so any number measured with it says
nothing about real dense retrieval. Reports produced with it are labelled SMOKE.
"""
from __future__ import annotations

import hashlib
import math
import re

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at", "is", "are", "we", "you", "our"}


class HashEmbedder:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.model = "hash-embedder-v1"

    def _vec(self, text: str) -> list[float]:
        toks = [t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str], kind: str = "doc") -> list[list[float]]:
        return [self._vec(t) for t in texts]
