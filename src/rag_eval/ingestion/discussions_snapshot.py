"""
Explicit, reproducible snapshot of GitHub Discussions Q&A pairs.

Unlike the docs corpus (pinned by commit SHA, see scripts/fetch_corpus.py),
Discussions are a live, growing feed -- `orderBy: CREATED_AT DESC` makes
"page 1" a sliding window over time, so an index and an eval set built from
separate live fetches taken on different days are silently comparing two
different corpora (docs/plan.md problem 3). This module makes the fetch a
deliberate, explicit step instead: a snapshot is a frozen list of discussion
ids + `fetched_at`, written to disk only when `fetch_snapshot` is run by
hand, never implicitly at ingest time.

Usage:
    uv run python -m rag_eval.ingestion.discussions_snapshot --max-pages 6
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rag_eval.common.schemas import DiscussionQA
from rag_eval.ingestion.github_discussions import fetch_discussion_qas

DEFAULT_SNAPSHOT_PATH = Path("data/corpus/discussions.json")


def fetch_snapshot(max_pages: int | None, out: Path = DEFAULT_SNAPSHOT_PATH) -> dict:
    """Fetch answered discussion Q&A pairs and write them to `out` as a
    frozen snapshot. Returns the snapshot dict that was written."""
    qas = fetch_discussion_qas(max_pages=max_pages)
    snapshot = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "max_pages": max_pages,
        "n_discussions": len(qas),
        "discussions": [qa.model_dump() for qa in qas],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def load_snapshot(path: Path = DEFAULT_SNAPSHOT_PATH) -> list[DiscussionQA]:
    """Load a previously written snapshot back into DiscussionQA records."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [DiscussionQA(**d) for d in data["discussions"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    args = parser.parse_args()

    snapshot = fetch_snapshot(args.max_pages, args.out)
    print(f"Wrote {snapshot['n_discussions']} discussions to {args.out}")
