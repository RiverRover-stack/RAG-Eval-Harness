"""Per-source corpus identity, stored in collection metadata so a stale
index is detectable without re-reading the whole corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SNAPSHOT_PATH = Path("data/corpus/SNAPSHOT.json")
DISCUSSIONS_PATH = Path("data/corpus/discussions.json")


def docs_corpus_sha(snapshot_path: Path = SNAPSHOT_PATH) -> str:
    """The pinned FastAPI commit SHA that data/corpus/docs was extracted from."""
    return json.loads(snapshot_path.read_text(encoding="utf-8"))["fastapi_sha"]


def discussions_corpus_sha(discussions_path: Path = DISCUSSIONS_PATH) -> str:
    """sha256 of the frozen discussions snapshot file, truncated to 12 hex chars --
    discussions have no upstream commit SHA to pin against, so the snapshot's own
    content hash is the corpus identity."""
    return hashlib.sha256(discussions_path.read_bytes()).hexdigest()[:12]
