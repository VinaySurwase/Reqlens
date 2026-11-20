from __future__ import annotations

import argparse
import asyncio
import json
import sys

from reqlens.db.session import make_session_factory
from reqlens.llm import build_embedder
from reqlens.observability.logging import new_run_id, setup_logging
from reqlens.pipeline import build_extractor, run_pipeline
from reqlens.sources.registry import ALL, build_adapter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reqlens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pipeline", help="ingest -> extract -> dedup -> index")
    p.add_argument("--sources", nargs="*", default=["greenhouse", "lever", "hn", "remotive"], choices=ALL)
    d = sub.add_parser("seed-demo", help="load SYNTHETIC fixtures through the full pipeline (offline providers)")
    d.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args(argv)
    setup_logging()
    new_run_id()
    factory = make_session_factory()
    with factory() as session:
        if a.cmd == "pipeline":
            res = asyncio.run(run_pipeline(session, [build_adapter(n) for n in a.sources],
                                           build_extractor(), build_embedder()))
        else:
            from reqlens.extract.heuristic import HeuristicExtractor
            from reqlens.index.embedder import HashEmbedder
            from reqlens.sources.fixture import FixtureServer
            srv = FixtureServer()
            res = asyncio.run(run_pipeline(session, list(srv.adapters().values()), HeuristicExtractor(),  # type: ignore[arg-type]
                                           HashEmbedder(), a.threshold))
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
