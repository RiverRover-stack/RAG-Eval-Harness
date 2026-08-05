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
  raw/               unprocessed pulls from GitHub (gitignored)
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

1. **Build the index** — pull Discussions, chunk, embed, upsert into Chroma:
   ```bash
   uv run python -m rag_eval.ingestion.embed_and_store
   ```
2. **Build the eval set** — separate from the index; ground truth comes
   straight from the accepted Discussion answers:
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

- Chunking is currently one chunk per answer; revisit if answers are long or
  code-heavy.
- RAGAS is using the same local Ollama model as both generator and judge —
  fine for iterating, but judge/generator overlap can inflate scores. Consider
  a stronger separate judge model before trusting absolute numbers.
- No retry/backoff on the GitHub GraphQL client yet; add if rate limits bite.
- `configs/` and `scripts/` are scaffolded but empty — fill in as needed.
