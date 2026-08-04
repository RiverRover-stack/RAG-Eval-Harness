"""
Glue script: fetch GitHub Discussions and load the FastAPI docs, chunk both,
and upsert them into their respective Chroma collections (fastapi_discussions
and fastapi_docs).

Usage:
    uv run python -m rag_eval.ingestion.embed_and_store
"""

from rag_eval.common.config import settings
from rag_eval.ingestion.chunker import qa_to_chunks
from rag_eval.ingestion.docs_chunker import load_doc_chunks
from rag_eval.ingestion.github_discussions import fetch_discussion_qas
from rag_eval.rag.vector_store import DISCUSSIONS_COLLECTION, DOCS_COLLECTION, upsert_chunks

DOCS_BATCH_SIZE = 64


def build_index(max_pages: int | None = None) -> int:
    """Fetch GitHub Discussions Q&A pairs, chunk, and upsert into fastapi_discussions."""
    qas = fetch_discussion_qas(max_pages=max_pages)
    total = 0
    for qa in qas:
        chunks = qa_to_chunks(qa)
        upsert_chunks(chunks, DISCUSSIONS_COLLECTION)
        total += len(chunks)
    return total


def build_docs_index(batch_size: int = DOCS_BATCH_SIZE, limit: int | None = None) -> int:
    """Load and chunk the FastAPI docs, and upsert into fastapi_docs.

    `limit` caps how many doc pages are indexed (see docs_chunker.load_doc_chunks).
    """
    total = 0
    batch: list[dict] = []
    for chunk in load_doc_chunks(limit=limit):
        batch.append(chunk)
        if len(batch) >= batch_size:
            upsert_chunks(batch, DOCS_COLLECTION)
            total += len(batch)
            batch = []
    if batch:
        upsert_chunks(batch, DOCS_COLLECTION)
        total += len(batch)
    return total


if __name__ == "__main__":
    # Both capped via INGEST_DISCUSSION_PAGES / INGEST_DOCS_LIMIT in .env,
    # so the retrieval corpus stays small enough for a single clean local run.
    n_discussions = build_index(max_pages=settings.ingest_discussion_pages or None)
    print(f"Indexed {n_discussions} chunks into Chroma ({DISCUSSIONS_COLLECTION})")

    n_docs = build_docs_index(limit=settings.ingest_docs_limit or None)
    print(f"Indexed {n_docs} chunks into Chroma ({DOCS_COLLECTION})")
