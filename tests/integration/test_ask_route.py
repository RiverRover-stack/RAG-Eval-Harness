"""Integration tests for POST /api/ask, ask.py's response shape, and the
new /api/stats and /api/runs routes -- via FastAPI's dependency_overrides
so no real Chroma collection, embedder, or LLM is touched (docs/plan.md's
constructor-injection-over-patching preference, adapted to FastAPI's own
override mechanism since routes pull the backend through `Depends`, not a
direct constructor call).
"""

from fastapi.testclient import TestClient

from rag_eval.api.deps import AppState, get_app_state
from rag_eval.api.main import app
from rag_eval.config.run_config import RunConfig
from rag_eval.providers.base import LLMResponse
from rag_eval.rag.prompts import PROMPTS
from rag_eval.retrieval.base import Candidate, RetrievalResult

client = TestClient(app)


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


def _override_state(pipeline: FakePipeline, llm: FakeLLM, fake_embedder) -> AppState:
    return AppState(
        cfg=RunConfig(name="test"),
        pipeline=pipeline,
        llm=llm,
        embedder=fake_embedder,
        prompt=PROMPTS["v2-cited"],
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
    assert body["citations"] == [{"index": 1, "chunk_id": "a", "url": "https://x/a"}]
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
