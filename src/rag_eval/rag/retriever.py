"""Thin retrieval interface on top of the Chroma vector store."""

from rag_eval.common.schemas import RetrievedChunk
from rag_eval.rag.vector_store import DISCUSSIONS_COLLECTION, DOCS_COLLECTION
from rag_eval.rag.vector_store import query as vector_query


def retrieve(question: str, k: int = 5) -> list[RetrievedChunk]:
    """Query both the discussions and docs collections, then keep the overall
    top-k hits by score. Both collections are embedded with the same Ollama
    model into the same cosine space, so their scores are directly comparable."""
    hits = vector_query(question, DISCUSSIONS_COLLECTION, k=k) + vector_query(
        question, DOCS_COLLECTION, k=k
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
