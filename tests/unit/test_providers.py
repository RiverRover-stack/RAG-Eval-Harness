import json
from typing import ClassVar, Self

import httpx
import pytest

from rag_eval.providers import get_embedder, get_llm
from rag_eval.providers.base import model_slug
from rag_eval.providers.embeddings.ollama import OllamaEmbedder
from rag_eval.providers.llm.gemini import GeminiLLM
from rag_eval.providers.llm.groq import GroqLLM
from rag_eval.providers.llm.ollama import OllamaLLM


class _FakeAsyncStreamResponse:
    """Fakes the bits of an httpx streaming response astream() touches:
    `raise_for_status()` and async-iterating lines. `status_code` lets a
    test simulate an upstream error surfacing mid-stream."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError(
                "error", request=request, response=httpx.Response(self.status_code, request=request)
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeAsyncStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncStreamResponse:
        return self._response

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeAsyncClient:
    """Fakes httpx.AsyncClient just enough for `async with httpx.AsyncClient(...) as
    client, client.stream(...) as response:` -- captures the call for
    assertions and hands back a scripted response."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    def stream(self, method: str, url: str, **kwargs):
        type(self).calls.append({"method": method, "url": url, **kwargs})
        return _FakeStreamContext(type(self)._next_response)


def _make_fake_async_client(lines: list[str], status_code: int = 200):
    """Returns a fresh _FakeAsyncClient subclass scripted with `lines`, and
    the shared `calls` list to assert against -- a fresh class per test so
    `calls` doesn't leak between tests."""

    class _Client(_FakeAsyncClient):
        calls: ClassVar[list[dict]] = []
        _next_response = _FakeAsyncStreamResponse(lines, status_code=status_code)

    return _Client


async def _collect(astream_call):
    return [chunk async for chunk in astream_call]


def test_model_slug_strips_org_prefix_and_dots():
    assert model_slug("BAAI/bge-small-en-v1.5") == "bge-small-en-v15"


def test_model_slug_passes_through_plain_model_name():
    assert model_slug("nomic-embed-text") == "nomic-embed-text"


def test_get_llm_is_cached_by_provider_and_model():
    a = get_llm("ollama", "fdm-llama")
    b = get_llm("ollama", "fdm-llama")
    c = get_llm("ollama", "some-other-model")
    assert a is b
    assert a is not c


def test_get_embedder_is_cached_by_provider_and_model():
    a = get_embedder("ollama", "nomic-embed-text")
    b = get_embedder("ollama", "nomic-embed-text")
    assert a is b


def test_get_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("not-a-real-provider", "x")


def test_get_embedder_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder("not-a-real-provider", "x")


def test_ollama_llm_complete_request_shape_and_response_parse(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(
            200,
            json={"message": {"content": "hi"}, "prompt_eval_count": 3, "eval_count": 5},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_eval.providers.llm.ollama.httpx.post", fake_post)

    llm = OllamaLLM(model="fdm-llama", base_url="http://localhost:11434")
    result = llm.complete([{"role": "user", "content": "hello"}], temperature=0.2, max_tokens=50)

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "fdm-llama"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["json"]["stream"] is False
    assert result.content == "hi"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 5


def test_groq_llm_complete_request_shape_and_response_parse(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_eval.providers.llm.groq.httpx.post", fake_post)

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="test-key")
    result = llm.complete([{"role": "user", "content": "hello"}])

    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "llama-3.3-70b-versatile"
    assert result.content == "answer"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4


def test_groq_llm_requires_api_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqLLM(model="llama-3.3-70b-versatile", api_key="")


def test_groq_llm_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("rag_eval.providers.llm.groq.time.sleep", lambda _seconds: None)
    calls = {"n": 0}

    def fake_post(url, *, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={}, request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}], "usage": {}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_eval.providers.llm.groq.httpx.post", fake_post)
    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="test-key")
    result = llm.complete([{"role": "user", "content": "hello"}])

    assert calls["n"] == 3
    assert result.content == "answer"


def test_groq_llm_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("rag_eval.providers.llm.groq.time.sleep", lambda _seconds: None)

    def always_429(url, *, headers, json, timeout):
        return httpx.Response(429, headers={}, request=httpx.Request("POST", url))

    monkeypatch.setattr("rag_eval.providers.llm.groq.httpx.post", always_429)
    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="test-key")

    with pytest.raises(httpx.HTTPStatusError):
        llm.complete([{"role": "user", "content": "hello"}])


def test_groq_llm_respects_retry_after_header(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("rag_eval.providers.llm.groq.time.sleep", sleeps.append)
    calls = {"n": 0}

    def fake_post(url, *, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "1.5"}, request=httpx.Request("POST", url)
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_eval.providers.llm.groq.httpx.post", fake_post)
    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="test-key")
    llm.complete([{"role": "user", "content": "hello"}])

    assert sleeps == [1.5]


def test_gemini_llm_complete_request_shape_and_response_parse(monkeypatch):
    captured = {}

    def fake_post(url, *, params, json, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["json"] = json
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "gemini answer"}]}}],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("rag_eval.providers.llm.gemini.httpx.post", fake_post)

    llm = GeminiLLM(model="gemini-2.5-flash", api_key="test-key")
    result = llm.complete(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hello"}]
    )

    assert captured["params"] == {"key": "test-key"}
    assert captured["json"]["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert captured["json"]["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert result.content == "gemini answer"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 2


def test_gemini_llm_requires_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLM(model="gemini-2.5-flash", api_key="")


async def test_groq_llm_astream_yields_deltas_then_a_done_chunk_with_usage(monkeypatch):
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello'}}]})}",
        f"data: {json.dumps({'choices': [{'delta': {'content': ' world'}}]})}",
        f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}})}",
        "data: [DONE]",
    ]
    fake_client_cls = _make_fake_async_client(lines)
    monkeypatch.setattr("rag_eval.providers.llm.groq.httpx.AsyncClient", fake_client_cls)

    llm = GroqLLM(model="openai/gpt-oss-120b", api_key="test-key")
    chunks = await _collect(llm.astream([{"role": "user", "content": "hi"}]))

    assert [c.delta for c in chunks] == ["Hello", " world", ""]
    assert chunks[-1].done is True
    assert chunks[-1].prompt_tokens == 10
    assert chunks[-1].completion_tokens == 2
    call = fake_client_cls.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["json"]["stream"] is True


async def test_groq_llm_astream_stops_at_done_sentinel(monkeypatch):
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'partial'}}]})}",
        "data: [DONE]",
        f"data: {json.dumps({'choices': [{'delta': {'content': 'should not appear'}}]})}",
    ]
    fake_client_cls = _make_fake_async_client(lines)
    monkeypatch.setattr("rag_eval.providers.llm.groq.httpx.AsyncClient", fake_client_cls)

    llm = GroqLLM(model="openai/gpt-oss-120b", api_key="test-key")
    chunks = await _collect(llm.astream([{"role": "user", "content": "hi"}]))

    assert [c.delta for c in chunks if c.delta] == ["partial"]


async def test_ollama_llm_astream_yields_deltas_then_a_done_chunk_with_usage(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " world"}, "done": False}),
        json.dumps(
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 4, "eval_count": 6}
        ),
    ]
    fake_client_cls = _make_fake_async_client(lines)
    monkeypatch.setattr("rag_eval.providers.llm.ollama.httpx.AsyncClient", fake_client_cls)

    llm = OllamaLLM(model="fdm-llama", base_url="http://localhost:11434")
    chunks = await _collect(llm.astream([{"role": "user", "content": "hi"}]))

    assert [c.delta for c in chunks] == ["Hello", " world", ""]
    assert chunks[-1].done is True
    assert chunks[-1].prompt_tokens == 4
    assert chunks[-1].completion_tokens == 6
    call = fake_client_cls.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["json"]["stream"] is True


async def test_gemini_llm_astream_raises_not_implemented():
    llm = GeminiLLM(model="gemini-2.5-flash", api_key="test-key")
    with pytest.raises(NotImplementedError):
        await _collect(llm.astream([{"role": "user", "content": "hi"}]))


def test_ollama_embedder_embed_query_parses_embedding(monkeypatch):
    def fake_post(url, *, json, timeout):
        assert json == {"model": "nomic-embed-text", "prompt": "hello"}
        return httpx.Response(
            200, json={"embedding": [0.1, 0.2, 0.3]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("rag_eval.providers.embeddings.ollama.httpx.post", fake_post)

    embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")
    assert embedder.embed_query("hello") == [0.1, 0.2, 0.3]
