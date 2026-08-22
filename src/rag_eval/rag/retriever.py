"""Thin retrieval interface on top of the Chroma vector store.

A back-compat wrapper (docs/plan.md Phase 6) over `retrieval.dense.DenseSearcher`
-- the naive cross-collection score merge this module used to do itself now
lives there, shared with the full `RetrievalPipeline`. Kept as a plain
dense-only, no-RunConfig call for callers that just want "the top-k chunks
for this question" without building a run config.
"""

from rag_eval.common.schemas import RetrievedChunk
from rag_eval.providers import get_embedder
from rag_eval.providers.base import EmbeddingProvider
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE
from rag_eval.rag.vector_store import query as vector_query
from rag_eval.retrieval.dense import DenseSearcher


def retrieve(
    question: str, k: int = 5, embedder: EmbeddingProvider | None = None
) -> list[RetrievedChunk]:
    """Embed the question once, query both the discussions and docs
    collections, then keep the overall top-k hits by score. Both are
    embedded with the same model into the same cosine space, so their
    scores are directly comparable."""
    embedder = embedder or get_embedder()
    searcher = DenseSearcher(
        sources=[DISCUSSIONS_SOURCE, DOCS_SOURCE], embedder=embedder, query_fn=vector_query
    )
    candidates = searcher.search(question, k)

    return [
        RetrievedChunk(content=c.content, source_id=c.url, score=c.final_score) for c in candidates
    ]
