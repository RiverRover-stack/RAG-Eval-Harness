"""Shared pytest fixtures.

`fake_embedder` and `ephemeral_collection` are pulled forward from Phase 4
(docs/plan.md) because Phase 2 needs them now: without a fake, unit tests
would instantiate real fastembed and download its ~90MB ONNX model at test
time, breaking CI's network-free guarantee. The rest of the Phase 4 fixture
set (sample_chunks, fake_llm, run_config, tmp_runs_dir) still isn't needed
until the eval runner lands.
"""

import hashlib

import chromadb
import pytest


class FakeEmbedder:
    """Deterministic, network-free stand-in for an EmbeddingProvider.
    Same text always hashes to the same dim-8 vector, so retrieval-stack
    tests can assert on ranking without touching a real model."""

    name = "fake"
    model = "fake-embedder"
    dim = 8
    slug = "fake"

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dim]]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def ephemeral_collection() -> chromadb.ClientAPI:
    """A real, in-memory Chroma client -- exercises the actual Chroma API
    (metadata semantics, query shape) without touching disk or a server.

    chromadb.EphemeralClient() instances share a process-global system cache
    keyed by settings (chromadb.api.client.SharedSystemClient), so two
    "separate" EphemeralClient() calls in the same test session silently see
    each other's collections. Clearing the cache before each client keeps
    tests isolated from each other.
    """
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    return chromadb.EphemeralClient()
