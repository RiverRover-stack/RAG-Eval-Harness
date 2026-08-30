"""POST /api/ask -- non-streaming ask endpoint (docs/plan.md Phase 7).

SSE streaming (`/api/ask/stream`) lands in a follow-up PR once `astream()`
exists on the LLM providers; this route retrieves, generates a v2-cited
answer, and returns the whole thing at once. The response shape mirrors the
plan's SSE `done` event ({answer, citations, groundedness, abstained,
usage, latency_ms}) on purpose, so the streaming route's final event can
reuse it.
"""

from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from rag_eval.api.deps import AppState, get_app_state
from rag_eval.common.telemetry import Usage, estimate_cost, log_request, timed
from rag_eval.rag.generator import generate_cited_answer

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    k: int | None = None


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    url: str


class UsageOut(BaseModel):
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[CitationOut]
    coverage: float
    groundedness: float
    abstained: bool
    usage: UsageOut
    latency_ms: float


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, state: AppState = Depends(get_app_state)) -> AskResponse:
    request_id = str(uuid.uuid4())
    try:
        with timed() as elapsed_ms:
            result = state.pipeline.retrieve(req.question, k=req.k)
            gen = generate_cited_answer(
                req.question,
                result.candidates,
                prompt=state.prompt,
                llm=state.llm,
                embedder=state.embedder,
                abstain_below=state.cfg.generation.groundedness.abstain_below,
                temperature=state.cfg.generation.llm.temperature,
                max_tokens=state.cfg.generation.llm.max_tokens,
            )
        latency_ms = elapsed_ms()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}") from e

    usage = Usage(
        provider=state.cfg.generation.llm.provider,
        model=state.cfg.generation.llm.model,
        prompt_tokens=gen.prompt_tokens or 0,
        completion_tokens=gen.completion_tokens or 0,
        latency_ms=latency_ms,
    )
    log_request(usage, request_id=request_id, endpoint="/api/ask", abstained=gen.abstained)

    return AskResponse(
        request_id=request_id,
        question=req.question,
        answer=gen.answer,
        citations=[
            CitationOut(index=c.index, chunk_id=c.chunk_id, url=c.url) for c in gen.citations.citations
        ],
        coverage=gen.citations.coverage,
        groundedness=gen.groundedness,
        abstained=gen.abstained,
        usage=UsageOut(
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            cost_usd=estimate_cost(usage),
        ),
        latency_ms=latency_ms,
    )
