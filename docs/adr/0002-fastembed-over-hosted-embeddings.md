# 2. fastembed over hosted embeddings

## Status

Accepted

## Context

Every retrieval query and every re-index needs an embedding call. A hosted
embeddings API (OpenAI, Cohere, a hosted Ollama) adds a network round trip
to the hot path, a per-call cost, and a dependency the demo goes down
without. The deploy target (docs/plan.md, "Locked decisions") is a single
Hugging Face Space container, so anything requiring a second running
service is a deployment liability, not just a latency one.

The alternative in-process options were `sentence-transformers` and
`fastembed`. `sentence-transformers` pulls `torch`, which roughly triples
the Docker image size (docs/plan.md: "breaks the single-container deploy
target") and CLAUDE.md already rules it out for the same reason on the
reranker side.

## Decision

Embed in-process with `fastembed`'s ONNX runtime, model
`BAAI/bge-small-en-v1.5` (384-dim). No torch, no server, no per-call
network cost. Verified before adopting: `fastembed` 0.8.0 resolves cleanly
against the project's existing `chromadb`-pinned `onnxruntime` 1.28.0,
`huggingface_hub` 1.26.0, and `tokenizers` 0.23.1 -- no conflicting pins,
no torch pulled in transitively.

Ollama remains available as a local-dev embedding backend
(`providers/embeddings/ollama.py`) purely so the pre-Phase-2
nomic-embed-text collections stay queryable for a nomic-vs-bge A/B; it is
not the default.

## Consequences

Index builds pay a one-time ~90MB model download (cached at
`settings.fastembed_cache_dir`) and ~30-60s of CPU encoding for the full
corpus. Runtime queries embed in milliseconds with no network call and no
per-query cost. Phase 3's container build bakes this cache into a Docker
layer specifically to avoid paying the download cost at cold start.
