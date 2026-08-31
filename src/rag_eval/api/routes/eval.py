"""Read-only run-artifact routes. Frontend consumer lands in Phase 9
(docs/plan.md: `/eval?run=<id>` fetching `/api/runs/{id}`) -- this is just
the API surface over runs/manifest.py's existing list_runs/load_run.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag_eval.runs.manifest import list_runs, load_run

router = APIRouter()


@router.get("/runs")
def get_runs() -> list[dict]:
    return [m.to_dict() for m in list_runs(include_pinned=True)]


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return load_run(run_id).to_dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
