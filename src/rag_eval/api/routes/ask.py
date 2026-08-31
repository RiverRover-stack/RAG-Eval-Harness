"""POST /api/ask (non-streaming) and POST /api/ask/stream (SSE),
docs/plan.md Phase 7.

/api/ask retrieves, generates a v2-cited answer, and returns the whole
thing at once. /api/ask/stream emits the same shape incrementally:

    event: meta       {request_id, config_hash, k}
    event: retrieval  {candidates:[...], timings:{...}}
    event: token      {"t": "FastAPI "}
    event: citation   {index, chunk_id, url, char_start, char_end}
    event: done        (same shape /api/ask returns)
    event: error       {detail}

Ordering is load-bearing: `retrieval` fires before the first `token`, so a
future trace panel can render while the answer is still streaming in. Both
routes share `score_answer` (rag/generator.py) so citation/groundedness
scoring behaves identically whether the answer arrived in one shot or in
deltas.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_eval.api.deps import AppState, get_app_state
from rag_eval.api.rate_limit import check_daily_budget, rate_limit
from rag_eval.common.telemetry import Usage, estimate_cost, log_request, timed
from rag_eval.config.run_config import config_hash
from rag_eval.rag.citations import StreamingCitationScanner
from rag_eval.rag.generator import generate_cited_answer, score_answer
from rag_eval.retrieval.base import Candidate

router = APIRouter(dependencies=[Depends(rate_limit), Depends(check_daily_budget)])


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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _candidate_out(c: Candidate) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "url": c.url,
        "title": c.title,
        "source_type": c.source_type,
        "scores": c.scores,
        "ranks": c.ranks,
        "stages": c.stages,
    }


async def _ask_stream_events(req: AskRequest, state: AppState) -> AsyncIterator[str]:
    request_id = str(uuid.uuid4())
    with timed() as elapsed_ms:
        yield _sse(
            "meta",
            {
                "request_id": request_id,
                "config_hash": config_hash(state.cfg),
                "k": req.k if req.k is not None else state.cfg.retrieval.top_k,
            },
        )

        try:
            result = state.pipeline.retrieve(req.question, k=req.k)
        except httpx.HTTPError as e:
            yield _sse("error", {"detail": f"RAG backend unavailable: {e}"})
            return

        yield _sse(
            "retrieval",
            {
                "candidates": [_candidate_out(c) for c in result.candidates],
                "timings": result.stage_timings,
            },
        )

        messages = [
            {"role": "system", "content": state.prompt.system_prompt},
            {"role": "user", "content": state.prompt.build_user_prompt(req.question, result.candidates)},
        ]
        scanner = StreamingCitationScanner(result.candidates)
        answer_parts: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            async for chunk in state.llm.astream(
                messages,
                temperature=state.cfg.generation.llm.temperature,
                max_tokens=state.cfg.generation.llm.max_tokens,
            ):
                if chunk.delta:
                    answer_parts.append(chunk.delta)
                    yield _sse("token", {"t": chunk.delta})
                    for citation in scanner.feed(chunk.delta):
                        yield _sse(
                            "citation",
                            {
                                "index": citation.index,
                                "chunk_id": citation.chunk_id,
                                "url": citation.url,
                                "char_start": citation.char_start,
                                "char_end": citation.char_end,
                            },
                        )
                if chunk.done:
                    prompt_tokens = chunk.prompt_tokens
                    completion_tokens = chunk.completion_tokens
        except httpx.HTTPError as e:
            yield _sse("error", {"detail": f"generation failed: {e}"})
            return

        answer = "".join(answer_parts)
        citations, groundedness, abstained = score_answer(
            answer, result.candidates, state.embedder, state.cfg.generation.groundedness.abstain_below
        )

    latency_ms = elapsed_ms()
    usage = Usage(
        provider=state.cfg.generation.llm.provider,
        model=state.cfg.generation.llm.model,
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        latency_ms=latency_ms,
    )
    log_request(usage, request_id=request_id, endpoint="/api/ask/stream", abstained=abstained)

    yield _sse(
        "done",
        {
            "request_id": request_id,
            "question": req.question,
            "answer": answer,
            "citations": [
                {"index": c.index, "chunk_id": c.chunk_id, "url": c.url} for c in citations.citations
            ],
            "coverage": citations.coverage,
            "groundedness": groundedness,
            "abstained": abstained,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": estimate_cost(usage),
            },
            "latency_ms": latency_ms,
        },
    )


@router.post("/ask/stream")
async def ask_stream(req: AskRequest, state: AppState = Depends(get_app_state)) -> StreamingResponse:
    return StreamingResponse(
        _ask_stream_events(req, state),
        media_type="text/event-stream",
        headers={
            # HF Spaces (and most proxies) buffer text/* responses by
            # default -- without this, the whole stream arrives at once
            # and the "live" trace panel never gets to render mid-stream.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
