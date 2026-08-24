# RAG + Eval Harness

A RAG pipeline over FastAPI's docs + GitHub Discussions. The primary metric
is a judge-free retrieval eval (recall/precision/nDCG/MRR against
URL-keyed gold documents, with bootstrap confidence intervals) — RAGAS is
kept as a secondary, generation-focused signal, not the primary trust
mechanism. See `docs/plan.md` for the phased build; `docs/adr/` for formal
architecture decisions.

## Stack

- Python 3.11, managed with `uv`
- Ingestion: GitHub GraphQL API + FastAPI docs, pinned to a committed
  snapshot (`data/corpus/`)
- Embeddings: `fastembed` (ONNX, in-process, no `torch`)
- Vector store: Chroma (local, persisted to `data/processed/chroma`),
  collections namespaced by `{source}__{embedder-slug}`
- Retrieval (eval path): dense + BM25 + RRF fusion + optional cross-encoder
  rerank + optional HyDE query rewrite + optional parent expansion,
  config-driven (`configs/`)
- LLM providers: pluggable (`groq`, `gemini`, `airforce`, `ollama`) via
  plain `httpx` — see `.env.example`
- API: FastAPI
- Eval: judge-free retrieval metrics (primary) + RAGAS (secondary,
  generation-focused)

## Project layout

```
src/rag_eval/
  ingestion/         corpus fetch, chunking, index building
  retrieval/          dense/BM25/fusion/rerank/HyDE/expand pipeline stages
  rag/                serving-path vector store, retriever, generator, pipeline
  eval/                eval-set construction, retrieval metrics, RAGAS, CI gate
  runs/                run manifests + leaderboard
  config/              RunConfig loading (yaml + extends + CLI overrides)
  providers/           embedding + LLM provider implementations
  api/                 FastAPI app
  common/              settings, shared pydantic schemas
  cli.py               `rag-eval` CLI entry point
data/
  corpus/              pinned FastAPI docs + discussions snapshot (tracked in git)
  raw/                 unused now that corpus/ is pinned (gitignored)
  processed/           chroma persistence dir (gitignored)
  eval_sets/           eval set JSONL (tracked in git)
runs/
  _pinned/             promoted run artifacts (tracked in git); everything else gitignored
tests/
  unit/                fast, no external services
  integration/          hits the FastAPI app / real services
configs/               RunConfig yaml (base + purpose configs + experiments/)
scripts/               one-off operational scripts
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # then fill in GITHUB_TOKEN and whichever LLM provider key(s) you need
```

fastembed's embedding model runs in-process — no local server required for
embeddings. LLM calls (generation, HyDE, RAGAS judge) go through whichever
provider you configure in `.env`.

## Workflow

0. **Pin the corpus** — one-time (or explicit refresh) fetch of the docs
   snapshot and the discussions snapshot; both are committed, so everyone
   indexes the same corpus:
   ```bash
   uv run python scripts/fetch_corpus.py
   uv run python -m rag_eval.ingestion.discussions_snapshot --max-pages 6
   ```
1. **Build the index**:
   ```bash
   make index   # uv run python -m rag_eval.ingestion.embed_and_store
   ```
2. **Run the judge-free retrieval eval** (primary metric):
   ```bash
   uv run rag-eval eval run --config configs/baseline.yaml
   ```
   Swap `--config` for any file under `configs/experiments/` to run an
   ablation; `uv run rag-eval runs pin <run_id>` promotes a run into
   `runs/_pinned/`.
3. **Run the API**:
   ```bash
   make serve   # uv run uvicorn rag_eval.api.main:app --reload
   ```
4. **Run the legacy RAGAS eval** (secondary signal):
   ```bash
   make eval   # uv run python -m rag_eval.eval.run_ragas
   ```

## Tests

```bash
make test   # uv run pytest -m "not slow and not llm"
```

## Open decisions / TODO

- The serving path (`api/main.py`) still retrieves via a naive dense-only
  merge; the full multi-stage retrieval pipeline (BM25/fusion/rerank/HyDE)
  exists and has been ablated, but no single stage combination was an
  unambiguous win on the eval sets, so it hasn't been promoted into serving
  yet.
- RAGAS is using the same model as both generator and judge in some
  configs — judge/generator overlap can inflate scores; a stronger,
  separate judge model is future work (Phase 8).
