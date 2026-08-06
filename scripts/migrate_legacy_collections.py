"""One-time rename: fastapi_docs / fastapi_discussions -> the namespaced
form (fastapi_docs__nomic-embed-text / fastapi_discussions__nomic-embed-text),
backfilling the embedding-identity metadata the new vector_store.py
asserts on read.

These collections predate embedding namespacing (Phase 1) and carry no
embedding_model/embedding_dim metadata. docs/plan.md says to "leave the old
collections in place" so a nomic-vs-bge A/B stays possible -- this script
is what makes that literal: same vectors, same chunk ids, just renamed and
labeled so the new assert-on-read path can address them. No re-embedding.

IMPORTANT: chromadb's Collection.modify(metadata=...) REPLACES the metadata
dict wholesale rather than merging -- verified empirically, not documented.
It also raises ValueError("Changing the distance function...") if
`hnsw:space` appears in that call at all, even set to its current value --
also verified empirically. Neither is a problem here: the distance
function lives in the collection's `configuration` (confirmed via a direct
query before/after a metadata-only modify()), not in the free-form
metadata dict, so leaving `hnsw:space` out of the metadata rewrite below
does not change it.

Usage:
    uv run python scripts/migrate_legacy_collections.py            # dry run
    uv run python scripts/migrate_legacy_collections.py --apply    # do it
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from rag_eval.common.config import settings
from rag_eval.ingestion.corpus_sha import discussions_corpus_sha, docs_corpus_sha
from rag_eval.rag.vector_store import get_client

LEGACY_EMBEDDING_MODEL = "nomic-embed-text"
LEGACY_EMBEDDING_DIM = 768

# (legacy collection name, source, corpus_sha getter)
MIGRATIONS = [
    ("fastapi_docs", "docs", docs_corpus_sha),
    ("fastapi_discussions", "discussions", discussions_corpus_sha),
]


def migrate(apply: bool) -> None:
    client = get_client(settings.chroma_persist_dir)
    existing = {c.name for c in client.list_collections()}

    for legacy_name, source, sha_getter in MIGRATIONS:
        new_name = f"{legacy_name}__nomic-embed-text"

        if legacy_name not in existing:
            print(f"[skip] {legacy_name!r} not found (already migrated or never built)")
            continue
        if new_name in existing:
            print(f"[skip] {new_name!r} already exists")
            continue

        collection = client.get_collection(name=legacy_name, embedding_function=None)
        count = collection.count()
        corpus_sha = sha_getter()
        metadata = {
            "embedding_model": LEGACY_EMBEDDING_MODEL,
            "embedding_dim": LEGACY_EMBEDDING_DIM,
            "corpus_sha": corpus_sha,
            "created_at": datetime.now(UTC).isoformat(),
        }

        print(f"{'[apply]' if apply else '[dry-run]'} {legacy_name!r} -> {new_name!r} "
              f"({count} chunks, corpus_sha={corpus_sha}, source={source})")

        if apply:
            collection.modify(name=new_name, metadata=metadata)
            print(f"  done. {new_name!r} now has metadata={collection.metadata}")

    if not apply:
        print("\nDry run only -- pass --apply to actually rename + backfill metadata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually perform the migration")
    args = parser.parse_args()
    migrate(apply=args.apply)
