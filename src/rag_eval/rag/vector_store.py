"""Chroma vector store wrapper, embedding via a local Ollama model."""

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from rag_eval.common.config import settings

DISCUSSIONS_COLLECTION = "fastapi_discussions"
DOCS_COLLECTION = "fastapi_docs"


def get_collection(collection_name: str):
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embed_fn = OllamaEmbeddingFunction(
        url=settings.ollama_base_url,
        model_name=settings.ollama_embed_model,
    )
    return client.get_or_create_collection(
        name=collection_name,
        # chromadb's EmbeddingFunction protocol and its own OllamaEmbeddingFunction
        # stub disagree on the accepted input type; a real mismatch in chromadb's
        # stubs, not ours. Resolved in Phase 2, which stops using Chroma's built-in
        # embedding functions entirely (embeddings computed and passed explicitly).
        embedding_function=embed_fn,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[dict], collection_name: str) -> None:
    """chunks: list of {id, document, metadata} from ingestion.chunker / docs_chunker."""
    collection = get_collection(collection_name)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def query(text: str, collection_name: str, k: int = 5) -> list[dict]:
    collection = get_collection(collection_name)
    result = collection.query(query_texts=[text], n_results=k)
    out = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        out.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return out
