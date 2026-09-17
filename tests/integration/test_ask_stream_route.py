"""Integration tests for POST /api/ask/stream (docs/plan.md Phase 7 SSE)
-- the plan's own testing table calls this the highest-value test: it
asserts event order `meta -> retrieval -> token+ -> done` and that every
`done.citations` entry validates.

Uses `FakeLLM.astream` (an async generator) via dependency_overrides, same
as test_ask_route.py -- no real Chroma, embedder, or provider touched.
"""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from rag_eval.api.deps import AppState, get_app_state
from rag_eval.api.main import app
from rag_eval.api.rate_limit import check_daily_budget, rate_limit
from rag_eval.config.run_config import RunConfig
from rag_eval.providers.base import StreamChunk
from rag_eval.rag.prompts import PROMPTS
from rag_eval.retrieval.base import Candidate, RetrievalResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def _bypass_rate_limit_and_budget():
    app.dependency_overrides[rate_limit] = lambda: None
    app.dependency_overrides[check_daily_budget] = lambda: None
    yield
    app.dependency_overrides.pop(rate_limit, None)
    app.dependency_overrides.pop(check_daily_budget, None)


class FakeStreamPipeline:
    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates

    def retrieve(self, query: str, *, k: int | None = None, deny_ids=frozenset()) -> RetrievalResult:
        return RetrievalResult(query=query, candidates=self._candidates)


class FakeStreamLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, deltas: list[str], prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self._deltas = deltas
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    async def astream(self, messages, *, temperature=0.0, max_tokens=1024) -> AsyncIterator[StreamChunk]:
        for delta in self._deltas:
            yield StreamChunk(delta=delta)
        yield StreamChunk(
            delta="", done=True, prompt_tokens=self._prompt_tokens, completion_tokens=self._completion_tokens
        )


def _candidate(chunk_id: str, content: str, url: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, content=content, url=url, title="t", source_type="docs")


def _override_state(pipeline, llm, fake_embedder) -> AppState:
    return AppState(
        cfg=RunConfig(name="test"),
        pipeline=pipeline,
        llm=llm,
        embedder=fake_embedder,
        prompt=PROMPTS["v2-cited"],
    )


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """[(event, data_json_str), ...], in order -- the wire format is
    `event: <name>\\ndata: <json>\\n\\n` per block."""
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((event, data))
    return events


def test_ask_stream_event_order_and_final_citations(fake_embedder):
    import json

    candidates = [_candidate("a", "FastAPI validates request bodies with Pydantic.", "https://x/a")]
    pipeline = FakeStreamPipeline(candidates)
    llm = FakeStreamLLM(["FastAPI validates ", "request bodies ", "with Pydantic [1]."])
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask/stream", json={"question": "How does FastAPI validate bodies?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    names = [name for name, _ in events]

    assert names[0] == "meta"
    assert names[1] == "retrieval"
    assert names[-1] == "done"
    assert all(name in {"meta", "retrieval", "token", "citation", "done"} for name in names)
    assert names.count("token") == 3  # one per delta fed to the LLM

    done = json.loads(events[-1][1])
    assert done["answer"] == "FastAPI validates request bodies with Pydantic [1]."
    assert done["citations"] == [
        {"index": 1, "chunk_id": "a", "url": "https://x/a", "path": "a", "gold": False}
    ]
    assert done["abstained"] is False
    assert done["usage"]["prompt_tokens"] == 10
    assert done["usage"]["completion_tokens"] == 5


def test_ask_stream_retrieval_event_precedes_first_token(fake_embedder):
    import json

    candidates = [_candidate("a", "content", "https://x/a")]
    pipeline = FakeStreamPipeline(candidates)
    llm = FakeStreamLLM(["hi"])
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask/stream", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    events = _parse_sse(resp.text)
    retrieval_idx = next(i for i, (name, _) in enumerate(events) if name == "retrieval")
    first_token_idx = next(i for i, (name, _) in enumerate(events) if name == "token")
    assert retrieval_idx < first_token_idx

    retrieval_payload = json.loads(events[retrieval_idx][1])
    assert retrieval_payload["candidates"][0]["chunk_id"] == "a"


def test_ask_stream_insufficient_context_reports_abstained(fake_embedder):
    import json

    candidates = [_candidate("a", "unrelated", "https://x/a")]
    pipeline = FakeStreamPipeline(candidates)
    llm = FakeStreamLLM(["INSUFFICIENT_CONTEXT: not covered"])
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask/stream", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    events = _parse_sse(resp.text)
    done = json.loads(events[-1][1])
    assert done["abstained"] is True
    assert done["citations"] == []


def test_ask_stream_sets_no_buffering_header(fake_embedder):
    candidates = [_candidate("a", "content", "https://x/a")]
    pipeline = FakeStreamPipeline(candidates)
    llm = FakeStreamLLM(["hi"])
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask/stream", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_ask_stream_emits_citation_event_split_across_deltas(fake_embedder):
    candidates = [_candidate("a", "content", "https://x/a")]
    pipeline = FakeStreamPipeline(candidates)
    # split the citation marker itself across two deltas -- proves the
    # route is really using StreamingCitationScanner, not re-scanning each
    # delta in isolation (which would miss a marker split like this).
    llm = FakeStreamLLM(["cited answer [", "1]."])
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask/stream", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    events = _parse_sse(resp.text)
    citation_events = [name for name, _ in events if name == "citation"]
    assert len(citation_events) == 1


def test_ask_stream_without_backend_returns_503():
    previous = getattr(app.state, "rag", None)
    app.state.rag = None
    try:
        resp = client.post("/api/ask/stream", json={"question": "does this need a live backend"})
    finally:
        app.state.rag = previous

    assert resp.status_code == 503
