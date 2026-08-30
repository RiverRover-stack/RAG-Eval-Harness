"""Health, readiness, and request-stats routes.

`/health` (bare, unprefixed) and `/api/health` are the pre-Phase-7 routes,
kept identical so test_static_mount.py's "static mount doesn't shadow the
API" assertion and any external uptime check against `/health` keep
working unchanged. `/api/health/ready` and `/api/stats` are unchanged from
their pre-Phase-7 shape too, just relocated out of api/main.py.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag_eval.common.telemetry import Stats, read_stats
from rag_eval.providers import get_embedder
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE, get_collection

# Unprefixed -- mounted directly on the app, not under the /api router.
bare_router = APIRouter()
router = APIRouter()


@bare_router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health")
def api_health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready() -> dict:
    """Fails loudly if the baked index is missing, empty, or embedded with
    a different model than the running embedder expects -- the Phase 3
    runtime guard (docs/plan.md): a cold container should 503, not silently
    serve empty retrievals."""
    embedder = get_embedder()
    problems: list[str] = []
    for source in (DOCS_SOURCE, DISCUSSIONS_SOURCE):
        try:
            collection = get_collection(source, embedder, create=False)
        except Exception as e:  # noqa: BLE001 - surfaced as a readiness detail, not swallowed
            problems.append(f"{source}: {e}")
            continue
        if collection.count() == 0:
            problems.append(f"{source}: collection is empty")
    if problems:
        raise HTTPException(status_code=503, detail={"ready": False, "problems": problems})
    return {"ready": True}


@router.get("/stats", response_model=Stats)
def stats() -> Stats:
    return read_stats()
