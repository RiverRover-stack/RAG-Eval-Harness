import httpx
import pytest

from rag_eval.ingestion import github_discussions as gd


def test_post_with_retry_succeeds_after_transient_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gd.time, "sleep", lambda _seconds: None)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        resp = gd._post_with_retry(client, url="https://example.com/graphql", json={})

    assert resp.status_code == 200
    assert calls["n"] == 3


def test_post_with_retry_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gd.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(gd, "MAX_RETRIES", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client, pytest.raises(httpx.HTTPStatusError):
        gd._post_with_retry(client, url="https://example.com/graphql", json={})


def test_post_with_retry_returns_immediately_on_success(monkeypatch: pytest.MonkeyPatch):
    sleeps = []
    monkeypatch.setattr(gd.time, "sleep", lambda seconds: sleeps.append(seconds))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        resp = gd._post_with_retry(client, url="https://example.com/graphql", json={})

    assert resp.status_code == 200
    assert sleeps == []
