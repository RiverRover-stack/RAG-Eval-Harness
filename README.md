# RAG + Eval Harness

A RAG pipeline over FastAPI's docs and GitHub Discussions Q&A, built around
a judge-free retrieval eval (recall/precision/nDCG/MRR against known-correct
gold documents) plus RAGAS as a secondary generation-quality signal — after
an audit found the original scaffold's RAGAS numbers reflected measurement
bugs, not retrieval quality. See `docs/plan.md` for the phased rebuild
driving ongoing work.

## Stack

- Python 3.11, managed with `uv`
- Ingestion: pinned, committed snapshots of FastAPI docs + GitHub
  Discussions (not a live re-fetch — see below)
- Vector store: Chroma (local, persisted), collections namespaced by
  `{source}__{embedder-slug}` so an embedder swap can't silently produce
  garbage results
- Embeddings: `fastembed` (ONNX, in-process, no `torch`)
- LLM providers: pluggable — Groq / Gemini / Ollama, via plain `httpx`
- API: FastAPI
- Eval config: strict, versioned `RunConfig` (`configs/*.yaml`)
- Eval metrics: judge-free retrieval metrics (recall@k, nDCG@k, MRR, ...)
  with bootstrap confidence intervals, plus RAGAS for generation quality
- CLI: `rag-eval` (eval runs, human review/labeling, run leaderboard)

## Project layout

```
src/rag_eval/
  ingestion/        corpus pinning, chunking, index building
  rag/               vector store, retriever, generator, pipeline
  providers/         LLM + embedding provider abstraction
  config/            RunConfig (yaml-based, extra="forbid")
  common/            env Settings, shared pydantic schemas
  eval/               eval-set construction, gold labels, retrieval
                       metrics, RAGAS scoring, CLI-backed review/labeling
  runs/               run manifests + leaderboard
  api/               FastAPI app
configs/             non-secret RunConfig yaml files (_base/baseline/ci/deploy)
data/
  corpus/            pinned FastAPI docs + discussions snapshot (tracked in git)
  eval_sets/         eval set JSONL files (tracked in git)
  processed/         Chroma persistence dir (gitignored)
runs/
  _pinned/           promoted run artifacts (tracked in git)
tests/
  unit/              fast, no external services
  integration/        hits the FastAPI app / real services
scripts/             one-off operational scripts (corpus fetch, migrations)
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # then fill in GITHUB_TOKEN
```

Requires a local Ollama server running with the models named in `.env`
(`ollama pull fdm-llama && ollama pull nomic-embed-text`, or swap in
whichever models you prefer), or a Groq/Gemini API key for those providers.

## Workflow

0. **Pin the corpus** — one-time (or explicit refresh) fetch of the docs
   snapshot and the discussions snapshot; both are committed, so everyone
   indexes the same corpus:
   ```bash
   uv run python scripts/fetch_corpus.py
   uv run python -m rag_eval.ingestion.discussions_snapshot --max-pages 6
   ```
1. **Build the index** — chunk both snapshots, embed, upsert into Chroma:
   ```bash
   uv run python -m rag_eval.ingestion.embed_and_store
   ```
2. **Run the eval**:
   ```bash
   # Judge-free retrieval metrics, versioned config
   uv run rag-eval eval run --config configs/baseline.yaml

   # RAGAS generation-quality scoring
   uv run python -m rag_eval.eval.run_ragas
   ```
3. **Run the API**:
   ```bash
   uv run uvicorn rag_eval.api.main:app --reload
   ```

Or via `make`: `make setup`, `make index`, `make eval`, `make serve`,
`make demo` (index + serve). See `Makefile` for the full list.

## Tests

```bash
uv run pytest -m "not slow and not llm"
```

## Current status

Corpus pinning, chunking fixes, the provider abstraction, and the
`RunConfig`/judge-free-metrics eval harness are implemented. Retrieval is
still a naive dense-vector merge (no fusion/reranking yet); the serving API
reads its LLM config from env vars rather than `RunConfig` yet; deploy is
moving from Hugging Face Spaces to Cloud Run (decision made, CD migration
in progress). Full status, results, and design rationale live in
`docs/PROJECT.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, and
`docs/EXPERIMENTS.md` (private, not committed) — see `CLAUDE.md` for
pointers if you're an agent working in this repo.
