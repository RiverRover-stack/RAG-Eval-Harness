# 3. Embedding-namespaced collections

## Status

Accepted

## Context

Phase 1 built `fastapi_docs` and `fastapi_discussions` with
`nomic-embed-text` (768-dim) via Chroma's built-in `OllamaEmbeddingFunction`.
Phase 2 switches the default embedder to `fastembed`'s
`BAAI/bge-small-en-v1.5` (384-dim, see ADR 0002). Querying a
768-dim-embedded collection with a 384-dim query vector doesn't cleanly
error inside Chroma's HNSW index -- it is exactly the kind of mismatch that
returns plausible-looking garbage instead of failing loudly, and a bare
`fastapi_docs` collection name carries no signal about which embedding
space it holds.

A second, independent need: docs/plan.md wants a nomic-vs-bge leaderboard
row (Phase 4), which requires both embedding spaces to coexist queryably at
once, not sequentially.

## Decision

Collection names are namespaced by embedder slug:
`fastapi_{source}__{embedder.slug}`, e.g. `fastapi_docs__bge-small-en-v15`.
Collection metadata additionally stores `{embedding_model, embedding_dim,
corpus_sha, created_at}`, written on creation and **asserted on every
read** (`vector_store._assert_embedder_matches`) -- a mismatch raises
`EmbedderMismatchError` rather than returning a nonsense-distance result.

The pre-existing nomic collections are migrated in place via
`scripts/migrate_legacy_collections.py`: renamed to
`fastapi_{source}__nomic-embed-text` and backfilled with the same metadata
block, with **no re-embedding** -- same vectors, same chunk ids, just
labeled. This was a deliberate deviation from re-indexing nomic separately:
it's zero-cost and gives the eventual A/B a cleaner single-variable
comparison, since both namespaces then hold the identical Phase-1-chunked
987+112 chunks.

Two collections created by different embedders are never assumed
comparable by row count or score scale alone; namespacing makes conflating
them require actively passing the wrong embedder object, not just typing
the wrong string.

## Consequences

Every embedding-model change is additive by construction -- switching
models never overwrites or requires deleting a prior index, at the cost of
Chroma disk space per embedding generation kept around (`chroma.sqlite3`
is gitignored, so this is only a local-disk cost, not a repo-size one).
`get_collection`'s `create=False` path is the one place a stale or
never-built collection surfaces as a clear `NotFoundError` /
`EmbedderMismatchError` rather than a silent wrong-space query.
