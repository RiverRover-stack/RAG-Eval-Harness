"""Chroma vector store wrapper: namespaced by embedding model, embeddings
computed by us and passed explicitly (never Chroma's built-in embedding
functions). See docs/adr/0003-embedding-namespaced-collections.md.

`source` is "docs" or "discussions"; the collection name is
"fastapi_{source}__{embedder.slug}", e.g. "fastapi_docs__bge-small-en-v15".
Old nomic-embedded collections are left in place under their own namespace
(migrated via scripts/migrate_legacy_collections.py) so a nomic-vs-bge A/B
stays possible without re-embedding.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import chromadb

from rag_eval.common.config import settings

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from rag_eval.providers.base import EmbeddingProvider

DOCS_SOURCE = "docs"
DISCUSSIONS_SOURCE = "discussions"


class EmbedderMismatchError(Exception):
    """Raised when a collection's stored embedding identity doesn't match
    the embedder being used to query or write it -- a mismatched-embedder
    query must fail loudly, not return garbage."""


@lru_cache
def get_client(persist_dir: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def collection_name(source: str, embedder: EmbeddingProvider) -> str:
    return f"fastapi_{source}__{embedder.slug}"


def _assert_embedder_matches(collection: Collection, embedder: EmbeddingProvider) -> None:
    meta = collection.metadata or {}
    stored_model = meta.get("embedding_model")
    stored_dim = meta.get("embedding_dim")
    if stored_model != embedder.model or stored_dim != embedder.dim:
        raise EmbedderMismatchError(
            f"Collection {collection.name!r} was embedded with "
            f"model={stored_model!r} dim={stored_dim!r}, but the requested "
            f"embedder is model={embedder.model!r} dim={embedder.dim!r}. "
            "Querying across a mismatched embedding space returns garbage, "
            "not an error -- this check exists to fail loudly instead."
        )


def get_collection(
    source: str,
    embedder: EmbeddingProvider,
    *,
    client: chromadb.ClientAPI | None = None,
    create: bool = False,
    corpus_sha: str | None = None,
) -> Collection:
    client = client or get_client(settings.chroma_persist_dir)
    name = collection_name(source, embedder)

    if create:
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedder.model,
                "embedding_dim": embedder.dim,
                "corpus_sha": corpus_sha or "",
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    else:
        collection = client.get_collection(name=name, embedding_function=None)

    _assert_embedder_matches(collection, embedder)
    return collection


def upsert_chunks(
    chunks: list[dict],
    source: str,
    embedder: EmbeddingProvider,
    *,
    client: chromadb.ClientAPI | None = None,
    corpus_sha: str | None = None,
    batch_size: int = 64,
) -> int:
    """chunks: list of {id, document, metadata} from ingestion.chunker / docs_chunker."""
    collection = get_collection(
        source, embedder, client=client, create=True, corpus_sha=corpus_sha
    )
    documents = [c["document"] for c in chunks]
    embeddings = embedder.embed_documents(documents, batch_size=batch_size)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=documents,
        # chromadb's stubs require an exact list[Sequence[float] | Sequence[int]]
        # type due to list invariance; list[list[float]] satisfies it structurally
        # at runtime but not under mypy's strict matching.
        embeddings=embeddings,  # type: ignore[arg-type]
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def query(
    embedding: list[float],
    source: str,
    embedder: EmbeddingProvider,
    k: int = 5,
    where: dict[str, Any] | None = None,
    *,
    client: chromadb.ClientAPI | None = None,
) -> list[dict]:
    collection = get_collection(source, embedder, client=client, create=False)
    result = collection.query(
        query_embeddings=[embedding],  # type: ignore[arg-type]
        n_results=k,
        where=where,
    )
    out = []
    # documents/metadatas/distances are declared Optional in chromadb's stubs but
    # are always present on a successful query() call.
    docs = result["documents"] or []
    metas = result["metadatas"] or []
    dists = result["distances"] or []
    for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
        out.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return out
