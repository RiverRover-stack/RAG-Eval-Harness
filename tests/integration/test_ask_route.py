"""Integration tests for POST /api/ask, ask.py's response shape, and the
new /api/stats and /api/runs routes -- via FastAPI's dependency_overrides
so no real Chroma collection, embedder, or LLM is touched (docs/plan.md's
constructor-injection-over-patching preference, adapted to FastAPI's own
override mechanism since routes pull the backend through `Depends`, not a
direct constructor call).

`rate_limit`/`check_daily_budget` are bypassed for every test in this file
by the autouse fixture below -- they share one process-global limiter
(api/rate_limit.py) across the whole test session, so leaving them live
here would make an unrelated test's pass/fail depend on how many other
tests already hit /api/ask* first. test_ask_returns_429_when_rate_limited
and test_ask_returns_402_when_daily_budget_exhausted below prove those
dependencies are actually attached to the router, without needing 10 real
requests to do it.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rag_eval.api.deps import AppState, get_app_state
from rag_eval.api.main import app
from rag_eval.api.rate_limit import check_daily_budget, rate_limit
from rag_eval.config.run_config import RunConfig
from rag_eval.providers.base import LLMResponse
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


class FakePipeline:
    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, k: int | None = None, deny_ids=frozenset()) -> RetrievalResult:
        self.calls.append((query, k))
        return RetrievalResult(query=query, candidates=self._candidates)


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, content: str) -> None:
        self._content = content

    def complete(self, messages, *, temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        return LLMResponse(content=self._content, model=self.model, prompt_tokens=10, completion_tokens=5)


def _candidate(chunk_id: str, content: str, url: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, content=content, url=url, title="t", source_type="docs")


def _override_state(pipeline: FakePipeline, llm: FakeLLM, fake_embedder, **extra) -> AppState:
    return AppState(
        cfg=RunConfig(name="test"),
        pipeline=pipeline,
        llm=llm,
        embedder=fake_embedder,
        prompt=PROMPTS["v2-cited"],
        **extra,
    )


def test_ask_returns_answer_with_citations_and_usage(fake_embedder):
    candidates = [_candidate("a", "FastAPI validates request bodies with Pydantic.", "https://x/a")]
    pipeline = FakePipeline(candidates)
    llm = FakeLLM("FastAPI validates request bodies with Pydantic [1].")
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask", json={"question": "How does FastAPI validate bodies?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "FastAPI validates request bodies with Pydantic [1]."
    assert body["citations"] == [
        {"index": 1, "chunk_id": "a", "url": "https://x/a", "path": "a", "gold": False}
    ]
    assert body["abstained"] is False
    assert body["usage"]["prompt_tokens"] == 10
    assert body["usage"]["completion_tokens"] == 5
    assert pipeline.calls == [("How does FastAPI validate bodies?", None)]


def test_ask_insufficient_context_reports_abstained(fake_embedder):
    candidates = [_candidate("a", "unrelated content", "https://x/a")]
    pipeline = FakePipeline(candidates)
    llm = FakeLLM("INSUFFICIENT_CONTEXT: rate limiting isn't covered")
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask", json={"question": "How do I rate-limit an endpoint?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_ask_rejects_empty_question(fake_embedder):
    pipeline = FakePipeline([_candidate("a", "content", "https://x/a")])
    llm = FakeLLM("answer")
    app.dependency_overrides[get_app_state] = lambda: _override_state(pipeline, llm, fake_embedder)
    try:
        resp = client.post("/api/ask", json={"question": ""})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 422


def test_ask_marks_citation_gold_only_for_a_matched_eval_item(fake_embedder):
    from rag_eval.eval.gold import EvalItem

    candidates = [_candidate("a", "FastAPI validates request bodies with Pydantic.", "https://x/a")]
    pipeline = FakePipeline(candidates)
    llm = FakeLLM("FastAPI validates request bodies with Pydantic [1].")
    item = EvalItem(
        id="1", dataset="docs_synth_v1", question="How does FastAPI validate bodies?", gold_chunk_ids=["a"]
    )
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        pipeline, llm, fake_embedder, eval_items_by_question={item.question: item}
    )
    try:
        resp = client.post("/api/ask", json={"question": "How does FastAPI validate bodies?"})
    finally:
        app.dependency_overrides.clear()

    body = resp.json()
    assert body["citations"] == [
        {"index": 1, "chunk_id": "a", "url": "https://x/a", "path": "a", "gold": True}
    ]


def test_ask_never_fabricates_gold_for_an_unmatched_question(fake_embedder):
    from rag_eval.eval.gold import EvalItem

    candidates = [_candidate("a", "FastAPI validates request bodies with Pydantic.", "https://x/a")]
    pipeline = FakePipeline(candidates)
    llm = FakeLLM("FastAPI validates request bodies with Pydantic [1].")
    # gold_chunk_ids=["a"] under a *different* question -- proves an
    # unmatched free-typed question never inherits another item's gold set.
    item = EvalItem(id="1", dataset="docs_synth_v1", question="some other question", gold_chunk_ids=["a"])
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        pipeline, llm, fake_embedder, eval_items_by_question={item.question: item}
    )
    try:
        resp = client.post("/api/ask", json={"question": "How does FastAPI validate bodies?"})
    finally:
        app.dependency_overrides.clear()

    body = resp.json()
    assert body["citations"][0]["gold"] is False


def test_ask_without_backend_returns_503():
    app.dependency_overrides.clear()
    previous = getattr(app.state, "rag", None)
    app.state.rag = None  # explicit, rather than relying on lifespan having failed to build one
    try:
        resp = client.post("/api/ask", json={"question": "does this need a live backend"})
    finally:
        app.state.rag = previous

    assert resp.status_code == 503


def test_stats_endpoint_is_not_empty_shaped():
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "request_count",
        "priced_request_count",
        "total_cost_usd",
        "p50_latency_ms",
        "p95_latency_ms",
    }


def test_runs_endpoint_returns_a_list():
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_run_detail_404s_for_unknown_run():
    resp = client.get("/api/runs/does-not-exist")
    assert resp.status_code == 404


def test_ask_returns_429_when_rate_limited(fake_embedder):
    def _raise_429():
        raise HTTPException(status_code=429, detail="rate limited")

    pipeline = FakePipeline([_candidate("a", "content", "https://x/a")])
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        pipeline, FakeLLM("answer"), fake_embedder
    )
    app.dependency_overrides[rate_limit] = _raise_429
    try:
        resp = client.post("/api/ask", json={"question": "hi"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 429


def test_ask_returns_402_when_daily_budget_exhausted(fake_embedder):
    def _raise_402():
        raise HTTPException(status_code=402, detail="budget exhausted")

    pipeline = FakePipeline([_candidate("a", "content", "https://x/a")])
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        pipeline, FakeLLM("answer"), fake_embedder
    )
    app.dependency_overrides[check_daily_budget] = _raise_402
    try:
        resp = client.post("/api/ask", json={"question": "hi"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 402


def test_feedback_logs_and_returns_ok(monkeypatch):
    import rag_eval.api.routes.ask as ask_route

    calls = []
    monkeypatch.setattr(
        ask_route, "log_feedback", lambda *, request_id, verdict: calls.append((request_id, verdict))
    )
    resp = client.post("/api/feedback", json={"request_id": "r1", "verdict": "good"})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert calls == [("r1", "good")]


def test_feedback_rejects_unknown_verdict():
    resp = client.post("/api/feedback", json={"request_id": "r1", "verdict": "meh"})
    assert resp.status_code == 422


def test_suggestions_returns_n_real_eval_questions(fake_embedder):
    from rag_eval.eval.gold import EvalItem

    items = {
        f"q{i}": EvalItem(id=str(i), dataset="docs_synth_v1", question=f"q{i}") for i in range(5)
    }
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        FakePipeline([]), FakeLLM("answer"), fake_embedder, eval_items_by_question=items
    )
    try:
        resp = client.get("/api/suggestions?n=3")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert set(body).issubset(items)


def test_suggestions_prefers_docs_synth_v1_over_discussions_v2(fake_embedder):
    from rag_eval.eval.gold import EvalItem

    items = {
        "docs-q": EvalItem(id="1", dataset="docs_synth_v1", question="docs-q"),
        "disc-q": EvalItem(id="2", dataset="discussions_v2", question="disc-q"),
    }
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        FakePipeline([]), FakeLLM("answer"), fake_embedder, eval_items_by_question=items
    )
    try:
        resp = client.get("/api/suggestions?n=1")
    finally:
        app.dependency_overrides.clear()

    assert resp.json() == ["docs-q"]


def test_suggestions_clamps_negative_n_instead_of_raising(fake_embedder):
    from rag_eval.eval.gold import EvalItem

    items = {"docs-q": EvalItem(id="1", dataset="docs_synth_v1", question="docs-q")}
    app.dependency_overrides[get_app_state] = lambda: _override_state(
        FakePipeline([]), FakeLLM("answer"), fake_embedder, eval_items_by_question=items
    )
    try:
        resp = client.get("/api/suggestions?n=-1")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json() == []
