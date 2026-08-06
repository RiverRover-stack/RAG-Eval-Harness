"""FastAPI app exposing the RAG pipeline, plus the built static frontend.

The frontend is `deploy/web-placeholder/` until Phase 9's Next.js export
replaces it (docs/plan.md). `StaticFiles` is mounted at "/" *last*, after
`/health` and the `/api` router are registered, so route order doesn't let
the catch-all static mount shadow the API -- see test_static_mount.py.
"""

import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_eval.providers import get_embedder, get_llm
from rag_eval.rag.pipeline import answer_question
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE, get_collection

# Until Phase 4's RunConfig makes the serving LLM a yaml value (docs/plan.md
# C1), the deploy container picks it via plain env vars rather than a
# Settings field -- Settings is reserved for things that cannot change a
# metric, and a Groq-vs-Ollama switch plainly can. Local dev (`make serve`)
# gets Ollama by default; the Docker image sets these explicitly.
SERVE_LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "ollama")
SERVE_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "fdm-llama")

STATIC_DIR = Path(os.getenv("STATIC_DIR", "deploy/web-placeholder"))

app = FastAPI(title="RAG Eval Harness API")
api_router = APIRouter(prefix="/api")


class AskRequest(BaseModel):
    question: str
    k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api_router.get("/health")
def api_health() -> dict:
    return {"status": "ok"}


@api_router.get("/health/ready")
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


@api_router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        llm = get_llm(SERVE_LLM_PROVIDER, SERVE_LLM_MODEL)
        result = answer_question(req.question, k=req.k, llm=llm)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}") from e
    return AskResponse(**result)


app.include_router(api_router)

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
