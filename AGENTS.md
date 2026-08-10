# AGENTS.md

Guidance for coding agents working in this repository.

## Project snapshot

This project is a RAG + evaluation harness over FastAPI docs and GitHub Discussions.
Primary focus is retrieval quality measurement, with generation scoring layered on top.

## Core workflow

```bash
make setup   # uv sync --extra dev
make lint    # uv run ruff check .
make type    # uv run mypy src
make test    # uv run pytest -m "not slow and not llm"
make index   # uv run python -m rag_eval.ingestion.embed_and_store
make eval    # uv run python -m rag_eval.eval.run_ragas
make serve   # uv run uvicorn rag_eval.api.main:app --reload
make demo    # index + serve
```

## Repository rules

- Treat `docs/plan.md` as the source of truth for phased rebuild work.
- If a metric can change behavior or outcomes, place it in `RunConfig` (`configs/*.yaml`), not environment settings.
- Keep `RunConfig` strict (`extra="forbid"`) and fail on unknown keys.
- Gold labels are URL-keyed, not chunk-id keyed.
- Guard retrieval evals against self-retrieval leakage using deny-lists when applicable.
- Report confidence intervals for metrics; avoid point-estimate-only reporting.
- Re-index only when embedding model changes; preserve collection namespacing for model A/B comparisons.
- Prefer constructor injection over patch-heavy tests for retrieval components.
- Prefer `fastembed` ONNX rerankers; avoid `sentence-transformers` dependency bloat.

## Testing expectations

- Run lint, typing, and relevant tests for any behavior change.
- Default fast test target is `pytest -m "not slow and not llm"`.
- Markers in use: `slow`, `integration`, `llm`.

## Data/layout notes

- `data/corpus/` is the pinned, committed corpus snapshot used for reproducible runs.
- `data/processed/` is local generated state (e.g., Chroma persistence).
- `data/raw/` is legacy and should not be treated as the canonical corpus source.

## Change discipline

- Make minimal, focused changes.
- Do not quietly remove caveats to inflate reported quality.
- Preserve reproducibility of evaluation inputs and outputs.
