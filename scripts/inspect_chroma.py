"""Inspect what's actually stored in the Chroma collection and sanity-check
the chunk schema against what chunker.py defines.

Usage:
    uv run python scripts/inspect_chroma.py
    uv run python scripts/inspect_chroma.py --collection fastapi_docs
    uv run python scripts/inspect_chroma.py --limit 10
    uv run python scripts/inspect_chroma.py --id <discussion_id>   # look up one chunk
"""

import argparse
import json

from rag_eval.rag.vector_store import DISCUSSIONS_COLLECTION, DOCS_COLLECTION, get_collection

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
        "chunk_index",
        "parent_id",
        "content_hash",
    },
}


def inspect(limit: int, lookup_id: str | None, collection_name: str) -> None:
    collection = get_collection(collection_name)
    count = collection.count()
    print(f"Collection: {collection.name}")
    print(f"Total chunks stored: {count}\n")

    if count == 0:
        print("Collection is empty, nothing to inspect. Run embed_and_store first.")
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
        "--collection",
        type=str,
        default=DISCUSSIONS_COLLECTION,
        choices=[DISCUSSIONS_COLLECTION, DOCS_COLLECTION],
        help="Which Chroma collection to inspect",
    )
    args = parser.parse_args()
    inspect(limit=args.limit, lookup_id=args.id, collection_name=args.collection)
