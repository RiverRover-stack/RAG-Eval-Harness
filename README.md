# RAG + Eval Harness

A RAG pipeline over FastAPI's GitHub Discussions Q&A, with a RAGAS-based eval harness.

## Stack

- Python 3.11, managed with `uv`
- Ingestion: GitHub GraphQL API (pulls answered Discussions as Q&A pairs)
- Vector store: Chroma (local, persisted to `data/processed/chroma`)
- LLM + embeddings: local via Ollama
- API: FastAPI
- Eval: RAGAS (faithfulness, answer relevancy, context precision, context recall)

## Project layout

```
src/rag_eval/
  ingestion/        GitHub GraphQL fetch, chunking, index building
  rag/               vector store, retriever, generator, pipeline
  eval/              eval-set construction, RAGAS scoring
  api/               FastAPI app
  common/            settings, shared pydantic schemas
data/
  corpus/            pinned FastAPI docs snapshot + discussions snapshot (tracked in git)
  raw/               unused now that corpus/ is pinned (gitignored)
  processed/         chroma persistence dir (gitignored)
  eval_sets/         eval set JSONL + ragas results (tracked in git)
tests/
  unit/              fast, no external services
  integration/        hits the FastAPI app / real services
configs/             non-secret config files (yaml/toml) if needed later
scripts/             one-off operational scripts
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # then fill in GITHUB_TOKEN
```

Requires a local Ollama server running with the models named in `.env`
(`ollama pull fdm-llama && ollama pull nomic-embed-text`, or swap in whichever
models you prefer).

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
2. **Build the eval set** — separate from the index; ground truth comes
   straight from the accepted Discussion answers in the pinned snapshot:
   ```bash
   uv run python -m rag_eval.eval.build_eval_set
   ```
3. **Run the API**:
   ```bash
   uv run uvicorn rag_eval.api.main:app --reload
   ```
4. **Run the eval**:
   ```bash
   uv run python -m rag_eval.eval.run_ragas
   ```

## Tests

```bash
uv run pytest
```

## Open decisions / TODO

- RAGAS is using the same local Ollama model as both generator and judge —
  fine for iterating, but judge/generator overlap can inflate scores. Consider
  a stronger separate judge model before trusting absolute numbers.
- `configs/` is scaffolded but empty — fill in as needed (Phase 4).
