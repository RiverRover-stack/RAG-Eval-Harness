# syntax=docker/dockerfile:1
#
# Four stages (docs/plan.md Phase 3):
#   web     -> placeholder static site (becomes the Next.js export in Phase 9)
#   deps    -> python dependencies via uv, frozen from uv.lock
#   index   -> bakes the Chroma index from the committed corpus snapshot
#              (see docs/adr/0005-bake-index-at-build-time.md)
#   runtime -> venv + baked index + static site, nothing else

FROM node:20-slim AS web
WORKDIR /web
COPY deploy/web-placeholder/ ./out/

FROM python:3.11-slim AS deps
RUN pip install --no-cache-dir uv
WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM deps AS index
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    CHROMA_PERSIST_DIR=/app/data/processed/chroma \
    FASTEMBED_CACHE_DIR=/app/data/processed/fastembed
# Read-only inputs: the committed corpus snapshot and discussions snapshot.
# No GITHUB_TOKEN here -- ingestion reads pinned snapshots, never live GitHub
# (docs/plan.md Phase 1).
COPY data/corpus/ data/corpus/
RUN python -m rag_eval.ingestion.embed_and_store

FROM python:3.11-slim AS runtime
RUN useradd --uid 1000 --create-home appuser
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY --from=index /app/data/processed /app/data/processed
COPY --from=index /app/data/corpus/SNAPSHOT.json data/corpus/SNAPSHOT.json
COPY --from=web /web/out /app/deploy/web-placeholder
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH" \
    CHROMA_PERSIST_DIR=/app/data/processed/chroma \
    FASTEMBED_CACHE_DIR=/app/data/processed/fastembed \
    STATIC_DIR=/app/deploy/web-placeholder \
    RAG_LLM_PROVIDER=groq \
    RAG_LLM_MODEL=llama-3.3-70b-versatile \
    PORT=7860
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["sh", "-c", "uvicorn rag_eval.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
