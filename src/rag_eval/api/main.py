"""FastAPI app factory (docs/plan.md Phase 7), plus the built static frontend.

The Docker image builds the real Next.js export (Phase 9) and serves it via
`STATIC_DIR=/app/deploy/web`. `deploy/web-placeholder/` is only the fallback
`STATIC_DIR` default below, for running bare `uvicorn` locally without a
Docker build and without a `STATIC_DIR` override. `StaticFiles` is mounted
at "/" *last*, after `/health` and the `/api` router are registered, so
route order doesn't let the catch-all static mount shadow the API -- see
test_static_mount.py.

The RAG backend (RunConfig -> RetrievalPipeline -> LLM/embedder/prompt,
api/deps.py) is built once in `lifespan`, not per-request, and warms the
embedder, reranker, and BM25 index at startup. Building it can fail (no
baked index yet, no LLM API key locally) -- that's swallowed here so the
container still comes up and `/api/health/ready` can report why, rather
than the whole app failing to start.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag_eval.api.deps import build_app_state
from rag_eval.api.routes import ask, eval, health
from rag_eval.common.config import settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(os.getenv("STATIC_DIR", "deploy/web-placeholder"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = os.getenv("RAG_CONFIG_PATH", settings.default_run_config)
    try:
        app.state.rag = build_app_state(config_path)
    except Exception:
        logger.exception(
            "RAG backend failed to initialize from %s -- serving degraded until fixed "
            "(see /api/health/ready)",
            config_path,
        )
        app.state.rag = None
    yield


def create_app(static_dir: Path = STATIC_DIR) -> FastAPI:
    app = FastAPI(title="RAG Eval Harness API", lifespan=lifespan)

    # Frontend dev server (`next dev`, :3000) talks to this API cross-origin
    # (:8000); the exported static build served from StaticFiles below is
    # same-origin and never needs this. Narrow allowlist, not "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.bare_router)

    api_router = APIRouter(prefix="/api")
    api_router.include_router(health.router)
    api_router.include_router(ask.router)
    api_router.include_router(eval.router)
    app.include_router(api_router)

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
