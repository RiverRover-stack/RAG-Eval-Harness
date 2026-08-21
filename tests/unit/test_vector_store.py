import pytest

from rag_eval.rag.vector_store import (
    EmbedderMismatchError,
    collection_name,
    get_collection,
    query,
    upsert_chunks,
)

CHUNKS = [
    {"id": "a", "document": "first chunk", "metadata": {"url": "https://x/a"}},
    {"id": "b", "document": "second chunk", "metadata": {"url": "https://x/b"}},
]


def test_collection_name_is_namespaced_by_embedder_slug(fake_embedder):
    assert collection_name("docs", fake_embedder) == "fastapi_docs__fake"
    assert collection_name("discussions", fake_embedder) == "fastapi_discussions__fake"


def test_get_collection_create_writes_embedding_identity_metadata(fake_embedder, ephemeral_collection):
    collection = get_collection(
        "docs", fake_embedder, client=ephemeral_collection, create=True, corpus_sha="deadbeef"
    )
    assert collection.metadata["embedding_model"] == fake_embedder.model
    assert collection.metadata["embedding_dim"] == fake_embedder.dim
    assert collection.metadata["corpus_sha"] == "deadbeef"
    assert collection.metadata["hnsw:space"] == "cosine"


def test_get_collection_create_refreshes_corpus_sha_on_reembed(fake_embedder, ephemeral_collection):
    # get_or_create_collection ignores `metadata` once the collection
    # already exists -- corpus_sha/created_at must be refreshed explicitly,
    # and embedding_model/embedding_dim must survive that refresh.
    first = get_collection(
        "docs", fake_embedder, client=ephemeral_collection, create=True, corpus_sha="sha-old"
    )
    created_at_first = first.metadata["created_at"]

    second = get_collection(
        "docs", fake_embedder, client=ephemeral_collection, create=True, corpus_sha="sha-new"
    )

    assert second.metadata["corpus_sha"] == "sha-new"
    assert second.metadata["created_at"] != created_at_first
    assert second.metadata["embedding_model"] == fake_embedder.model
    assert second.metadata["embedding_dim"] == fake_embedder.dim
    # hnsw:space itself can't survive a modify() call (chromadb rejects
    # re-specifying it even unchanged), but the actual distance function is
    # tracked separately in the collection's configuration, not metadata.
    assert second.configuration_json["hnsw"]["space"] == "cosine"


def test_upsert_then_query_roundtrips_and_computes_cosine_score(fake_embedder, ephemeral_collection):
    upsert_chunks(CHUNKS, "docs", fake_embedder, client=ephemeral_collection, corpus_sha="sha")

    query_embedding = fake_embedder.embed_query("first chunk")
    hits = query(query_embedding, "docs", fake_embedder, k=2, client=ephemeral_collection)

    assert len(hits) == 2
    assert hits[0]["content"] == "first chunk"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert hits[0]["metadata"]["url"] == "https://x/a"


def test_query_where_passthrough(fake_embedder, ephemeral_collection):
    upsert_chunks(CHUNKS, "docs", fake_embedder, client=ephemeral_collection, corpus_sha="sha")

    query_embedding = fake_embedder.embed_query("second chunk")
    hits = query(
        query_embedding,
        "docs",
        fake_embedder,
        k=5,
        where={"url": "https://x/b"},
        client=ephemeral_collection,
    )

    assert len(hits) == 1
    assert hits[0]["metadata"]["url"] == "https://x/b"


def test_get_collection_raises_on_embedder_mismatch(fake_embedder, ephemeral_collection):
    """A collection embedded with one model, queried with another, must
    fail loudly rather than return nonsense-distance results."""
    ephemeral_collection.get_or_create_collection(
        name="fastapi_docs__fake",
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "some-other-model",
            "embedding_dim": 999,
            "corpus_sha": "sha",
        },
    )

    with pytest.raises(EmbedderMismatchError):
        get_collection("docs", fake_embedder, client=ephemeral_collection, create=False)


def test_get_collection_read_missing_collection_raises(fake_embedder, ephemeral_collection):
    with pytest.raises(Exception):  # noqa: B017 - chromadb's own NotFoundError
        get_collection("docs", fake_embedder, client=ephemeral_collection, create=False)
