"""
Glue script: load the pinned discussions snapshot and the FastAPI docs
corpus, chunk both, and upsert them into their respective Chroma
collections (fastapi_discussions and fastapi_docs).

Both sources are read from committed snapshots -- data/corpus/discussions.json
(ingestion/discussions_snapshot.py) and data/corpus/docs + docs_src
(scripts/fetch_corpus.py) -- not fetched live. Refreshing either is an
explicit, separate step; there are no ingest_docs_limit/ingest_discussion_pages
truncation knobs here anymore (docs/plan.md problem 1: a 30-page alphabetical
cap silently excluded every `tutorial/` page).

Usage:
    uv run python -m rag_eval.ingestion.embed_and_store
"""

from rag_eval.ingestion.chunker import qa_to_chunks
from rag_eval.ingestion.discussions_snapshot import DEFAULT_SNAPSHOT_PATH, load_snapshot
from rag_eval.ingestion.docs_chunker import load_doc_chunks
from rag_eval.rag.vector_store import DISCUSSIONS_COLLECTION, DOCS_COLLECTION, upsert_chunks

DOCS_BATCH_SIZE = 64


def build_index(snapshot_path=DEFAULT_SNAPSHOT_PATH) -> int:
    """Chunk and upsert every discussion Q&A in the pinned snapshot into fastapi_discussions."""
    qas = load_snapshot(snapshot_path)
    total = 0
    for qa in qas:
        chunks = qa_to_chunks(qa)
        upsert_chunks(chunks, DISCUSSIONS_COLLECTION)
        total += len(chunks)
    return total


def build_docs_index(batch_size: int = DOCS_BATCH_SIZE) -> int:
    """Load, chunk, and upsert every doc page in the committed corpus into fastapi_docs."""
    total = 0
    batch: list[dict] = []
    for chunk in load_doc_chunks():
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
    n_discussions = build_index()
    print(f"Indexed {n_discussions} chunks into Chroma ({DISCUSSIONS_COLLECTION})")

    n_docs = build_docs_index()
    print(f"Indexed {n_docs} chunks into Chroma ({DOCS_COLLECTION})")
