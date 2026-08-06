# 5. Bake the Chroma index at Docker build time

## Status

Accepted

## Context

The deploy target (docs/plan.md, "Locked decisions") is a single Hugging
Face Space container on the free tier, which sleeps after inactivity and
cold-starts on the next request. If the container built its Chroma index on
startup -- chunking the committed corpus, embedding ~1200 chunks with
fastembed -- that 30-60s of CPU-bound work would sit in front of every cold
start. A first-time visitor's request would look hung or broken well before
it reached the model.

Embedding at request time was never on the table for the same reason
ADR 0002 rules out a hosted embeddings API: it adds latency and a failure
mode to the one path every user exercises. The index only needs to exist by
the time the container starts serving traffic, and the corpus and chunker
are already deterministic (docs/plan.md Phase 1: `SNAPSHOT.json`,
`discussions.json`) -- nothing about the index depends on when the
container happens to boot.

## Decision

The Dockerfile's `index` stage runs `embed_and_store.build_index` /
`build_docs_index` against the committed `data/corpus/` snapshot during
`docker build`, and the `runtime` stage copies the resulting
`data/processed/chroma` (sqlite + vectors) and the warmed fastembed ONNX
cache in as image layers. No network calls happen at container start, and
`GITHUB_TOKEN` is never present at runtime -- ingestion only needs it to
*refresh* a snapshot, an explicit, separate step (docs/plan.md Phase 1),
never to serve.

`/api/health/ready` asserts the baked collections exist, are non-empty, and
carry the embedding identity the running embedder expects
(`EmbedderMismatchError`, ADR 0003) -- a build that silently produced an
empty or mismatched index fails loudly at the health check instead of
serving empty retrievals.

## Consequences

Every corpus or embedding-model change requires a full image rebuild rather
than a config toggle -- acceptable because both are already rare, explicit
events (docs/plan.md: "Index the corpus exactly once per embedding model
change"). The image is bigger (~1.8MB of vectors, ~15MB sqlite, ~90MB
fastembed ONNX cache baked in) in exchange for a byte-identical index across
restarts and a cold start bounded by uvicorn startup, not by re-embedding
~1200 chunks on a shared build CPU.
