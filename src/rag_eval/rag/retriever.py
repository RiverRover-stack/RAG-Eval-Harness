"""Thin retrieval interface on top of the Chroma vector store."""

from rag_eval.common.schemas import RetrievedChunk
from rag_eval.providers import get_embedder
from rag_eval.providers.base import EmbeddingProvider
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE
from rag_eval.rag.vector_store import query as vector_query


def retrieve(
    question: str, k: int = 5, embedder: EmbeddingProvider | None = None
) -> list[RetrievedChunk]:
    """Embed the question once, query both the discussions and docs
    collections, then keep the overall top-k hits by score. Both are
    embedded with the same model into the same cosine space, so their
    scores are directly comparable."""
    embedder = embedder or get_embedder()
    query_embedding = embedder.embed_query(question)

    hits = vector_query(query_embedding, DISCUSSIONS_SOURCE, embedder, k=k) + vector_query(
        query_embedding, DOCS_SOURCE, embedder, k=k
    )
    hits.sort(key=lambda hit: hit["score"], reverse=True)

    return [
        RetrievedChunk(
            content=hit["content"],
            source_id=hit["metadata"].get("url", ""),
            score=hit["score"],
        )
        for hit in hits[:k]
    ]
