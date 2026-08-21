"""Inspect what's actually stored in a namespaced Chroma collection and
sanity-check the chunk schema against what chunker.py defines.

Usage:
    uv run python scripts/inspect_chroma.py
    uv run python scripts/inspect_chroma.py --source docs --embedder-model BAAI/bge-small-en-v1.5
    uv run python scripts/inspect_chroma.py --limit 10
    uv run python scripts/inspect_chroma.py --id <discussion_id>   # look up one chunk
"""

import argparse
import json

from rag_eval.providers import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER, get_embedder
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE, get_collection

# Metadata keys each source's chunker actually produces, keyed by source_type.
# discussion: chunker.py's qa_to_chunks(). docs: docs_chunker.py's doc_to_chunks().
EXPECTED_METADATA_KEYS_BY_SOURCE = {
    "discussion": {
        "source_type",
        "title",
        "question",
        "url",
        "category",
        "chunk_index",
        "parent_id",
        "content_hash",
    },
    "docs": {
        "source_type",
        "title",
        "section",
        "path",
        "url",
        "merged_urls",
        "chunk_index",
        "parent_id",
        "content_hash",
    },
}


def inspect(
    limit: int, lookup_id: str | None, source: str, embedder_provider: str, embedder_model: str
) -> None:
    embedder = get_embedder(embedder_provider, embedder_model)
    collection = get_collection(source, embedder, create=False)
    count = collection.count()
    print(f"Collection: {collection.name}")
    print(f"Embedding: {collection.metadata.get('embedding_model')} (dim={collection.metadata.get('embedding_dim')})")
    print(f"Corpus SHA: {collection.metadata.get('corpus_sha')}")
    print(f"Total chunks stored: {count}\n")

    if count == 0:
        print("Collection is empty, nothing to inspect. Run `make index` first.")
        return

    if lookup_id:
        result = collection.get(ids=[lookup_id], include=["documents", "metadatas"])
        ids = result["ids"]
    else:
        result = collection.get(limit=limit, include=["documents", "metadatas"])
        ids = result["ids"]

    if not ids:
        print("No matching records found.")
        return

    seen_key_sets = set()

    for i, chunk_id in enumerate(ids):
        doc = result["documents"][i]
        meta = result["metadatas"][i]
        meta_keys = frozenset(meta.keys())
        seen_key_sets.add(meta_keys)

        print(f"--- chunk {i + 1} ---")
        print(f"id: {chunk_id}")
        print(f"document (first 200 chars): {doc[:200]!r}")
        print(f"metadata: {json.dumps(meta, indent=2)}")

        source_type = meta.get("source_type")
        expected = EXPECTED_METADATA_KEYS_BY_SOURCE.get(source_type)
        if expected is None:
            print(f"Unknown source_type {source_type!r}, no schema to check against.")
        else:
            missing = expected - meta_keys
            extra = meta_keys - expected
            if missing:
                print(f"MISSING expected keys for source_type={source_type!r}: {sorted(missing)}")
            if extra:
                print(f"Extra/unexpected keys for source_type={source_type!r}: {sorted(extra)}")
            if not missing and not extra:
                print(f"Schema matches source_type={source_type!r} chunker exactly.")
        print()

    print("=== Summary ===")
    print(f"Inspected {len(ids)} chunk(s).")
    print(f"Found {len(seen_key_sets)} distinct metadata key-set(s) across inspected chunks:")
    for ks in seen_key_sets:
        print(f"  {sorted(ks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of chunks to sample")
    parser.add_argument("--id", type=str, default=None, help="Look up one chunk by id")
    parser.add_argument(
        "--source",
        type=str,
        default=DISCUSSIONS_SOURCE,
        choices=[DISCUSSIONS_SOURCE, DOCS_SOURCE],
        help="Which source's collection to inspect",
    )
    parser.add_argument("--embedder-provider", type=str, default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedder-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    inspect(
        limit=args.limit,
        lookup_id=args.id,
        source=args.source,
        embedder_provider=args.embedder_provider,
        embedder_model=args.embedder_model,
    )
