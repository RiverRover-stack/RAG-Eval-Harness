"""
Glue script: load the pinned discussions snapshot and the FastAPI docs
corpus, chunk both, and upsert them into their namespaced Chroma
collections (fastapi_discussions__<slug> and fastapi_docs__<slug>).

Both sources are read from committed snapshots -- data/corpus/discussions.json
(ingestion/discussions_snapshot.py) and data/corpus/docs + docs_src
(scripts/fetch_corpus.py) -- not fetched live. Refreshing either is an
explicit, separate step; there are no ingest_docs_limit/ingest_discussion_pages
truncation knobs here anymore (docs/plan.md problem 1: a 30-page alphabetical
cap silently excluded every `tutorial/` page).

Embeddings come from providers.get_embedder() (default: fastembed
BAAI/bge-small-en-v1.5) -- see docs/adr/0002-fastembed-over-hosted-embeddings.md.

Usage:
    uv run python -m rag_eval.ingestion.embed_and_store
"""

from rag_eval.ingestion.chunker import qa_to_chunks
from rag_eval.ingestion.corpus_sha import discussions_corpus_sha, docs_corpus_sha
from rag_eval.ingestion.discussions_snapshot import DEFAULT_SNAPSHOT_PATH, load_snapshot
from rag_eval.ingestion.docs_chunker import load_doc_chunks
from rag_eval.providers import get_embedder
from rag_eval.providers.base import EmbeddingProvider
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE, upsert_chunks

DOCS_BATCH_SIZE = 64


def build_index(
    embedder: EmbeddingProvider | None = None, snapshot_path=DEFAULT_SNAPSHOT_PATH
) -> int:
    """Chunk and upsert every discussion Q&A in the pinned snapshot into fastapi_discussions."""
    embedder = embedder or get_embedder()
    corpus_sha = discussions_corpus_sha()
    qas = load_snapshot(snapshot_path)
    total = 0
    for qa in qas:
        chunks = qa_to_chunks(qa)
        upsert_chunks(chunks, DISCUSSIONS_SOURCE, embedder, corpus_sha=corpus_sha)
        total += len(chunks)
    return total


def build_docs_index(
    embedder: EmbeddingProvider | None = None, batch_size: int = DOCS_BATCH_SIZE
) -> int:
    """Load, chunk, and upsert every doc page in the committed corpus into fastapi_docs."""
    embedder = embedder or get_embedder()
    corpus_sha = docs_corpus_sha()
    total = 0
    batch: list[dict] = []
    for chunk in load_doc_chunks():
        batch.append(chunk)
        if len(batch) >= batch_size:
            upsert_chunks(batch, DOCS_SOURCE, embedder, corpus_sha=corpus_sha)
            total += len(batch)
            batch = []
    if batch:
        upsert_chunks(batch, DOCS_SOURCE, embedder, corpus_sha=corpus_sha)
        total += len(batch)
    return total


if __name__ == "__main__":
    embedder = get_embedder()
    print(f"Embedding with {embedder.name}/{embedder.model} (dim={embedder.dim})")

    n_discussions = build_index(embedder)
    print(f"Indexed {n_discussions} chunks into Chroma (fastapi_discussions__{embedder.slug})")

    n_docs = build_docs_index(embedder)
    print(f"Indexed {n_docs} chunks into Chroma (fastapi_docs__{embedder.slug})")
