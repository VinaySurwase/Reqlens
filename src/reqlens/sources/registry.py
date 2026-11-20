from __future__ import annotations

from pathlib import Path

import yaml

from reqlens.config import get_settings
from reqlens.sources.base import SourceAdapter
from reqlens.sources.greenhouse import GreenhouseAdapter
from reqlens.sources.hn import HNWhoIsHiringAdapter
from reqlens.sources.lever import LeverAdapter
from reqlens.sources.remotive import RemotiveAdapter

ALL = ["greenhouse", "lever", "hn", "remotive"]


def load_boards(path: str | None = None) -> dict[str, list[dict[str, str]]]:
    p = Path(path or get_settings().boards_file)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return {"greenhouse": (data or {}).get("greenhouse") or [], "lever": (data or {}).get("lever") or []}


def build_adapter(name: str) -> SourceAdapter:
    boards = load_boards()
    if name == "greenhouse":
        return GreenhouseAdapter(boards["greenhouse"])
    if name == "lever":
        return LeverAdapter(boards["lever"])
    if name == "hn":
        return HNWhoIsHiringAdapter()
    if name == "remotive":
        return RemotiveAdapter()
    raise ValueError(f"unknown source {name!r}; choose from {ALL}")
