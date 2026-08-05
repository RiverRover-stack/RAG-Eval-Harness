import hashlib
import json
from pathlib import Path

import pytest

from rag_eval.ingestion.docs_chunker import load_doc_chunks

SNAPSHOT_PATH = Path("data/corpus/SNAPSHOT.json")


@pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="corpus not fetched yet -- run scripts/fetch_corpus.py"
)
def test_snapshot_chunk_count_matches_a_fresh_chunker_run():
    """Regression test for docs/plan.md problem 3: a chunker change (or a
    stale/mismatched corpus) that silently shifts the chunk set should fail
    here instead of silently shipping a different retrieval corpus than
    what SNAPSHOT.json records."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    chunk_ids = sorted(c["id"] for c in load_doc_chunks())
    chunk_count_hash = hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()[:16]

    assert len(chunk_ids) == snapshot["n_chunks_expected"]
    assert chunk_count_hash == snapshot["chunk_count_hash"]
