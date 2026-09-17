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
import random
import uuid
from collections.abc import AsyncIterator
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_eval.api.deps import AppState, get_app_state
from rag_eval.api.rate_limit import check_daily_budget, rate_limit
from rag_eval.common.telemetry import Usage, estimate_cost, log_feedback, log_request, timed
from rag_eval.config.run_config import config_hash
from rag_eval.rag.citations import Citation, StreamingCitationScanner
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
    path: str
    gold: bool


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


def _display_path(url: str) -> str:
    """Short slug for a source tile's label, e.g. `tutorial/dependencies` or
    `dependencies-with-yield` -- path plus fragment when both are present,
    whichever half exists otherwise, falling back to the full url."""
    parts = urlsplit(url)
    path = parts.path.strip("/")
    if parts.fragment:
        path = f"{path}#{parts.fragment}" if path else parts.fragment
    return path or url


def _gold_chunk_ids(question: str, state: AppState) -> set[str]:
    """Only ever populated for a question that exactly matches a loaded eval
    item -- a free-typed question always gets an empty set, never a
    fabricated gold match."""
    item = state.eval_items_by_question.get(question.strip())
    return set(item.gold_chunk_ids) if item else set()


def _citation_payload(
    citation: Citation, candidates_by_id: dict[str, Candidate], gold_chunk_ids: set[str]
) -> dict:
    candidate = candidates_by_id.get(citation.chunk_id)
    return {
        "index": citation.index,
        "chunk_id": citation.chunk_id,
        "url": citation.url,
        "path": _display_path(candidate.url if candidate else citation.url),
        "gold": citation.chunk_id in gold_chunk_ids,
    }


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

    candidates_by_id = {c.chunk_id: c for c in result.candidates}
    gold_chunk_ids = _gold_chunk_ids(req.question, state)
    return AskResponse(
        request_id=request_id,
        question=req.question,
        answer=gen.answer,
        citations=[
            CitationOut(**_citation_payload(c, candidates_by_id, gold_chunk_ids))
            for c in gen.citations.citations
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

    candidates_by_id = {c.chunk_id: c for c in result.candidates}
    gold_chunk_ids = _gold_chunk_ids(req.question, state)
    yield _sse(
        "done",
        {
            "request_id": request_id,
            "question": req.question,
            "answer": answer,
            "citations": [
                _citation_payload(c, candidates_by_id, gold_chunk_ids) for c in citations.citations
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


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., max_length=64)
    verdict: Literal["good", "bad"]


@router.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    log_feedback(request_id=req.request_id, verdict=req.verdict)
    return {"ok": True}


@router.get("/suggestions", response_model=list[str])
def suggestions(n: int = 3, state: AppState = Depends(get_app_state)) -> list[str]:
    """Real, gold-resolvable eval-set questions for the "Try" chips --
    docs_synth_v1 preferred, topped up with discussions_v2 if there aren't
    enough (see EVAL_DATASETS in api/deps.py for the load order)."""
    n = max(0, n)
    preferred = [
        q for q, item in state.eval_items_by_question.items() if item.dataset == "docs_synth_v1"
    ]
    rest = [q for q, item in state.eval_items_by_question.items() if item.dataset != "docs_synth_v1"]
    pool = preferred if len(preferred) >= n else preferred + rest
    return random.sample(pool, min(n, len(pool)))
