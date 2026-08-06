from fastapi.testclient import TestClient

from rag_eval.api.main import app

client = TestClient(app)


def test_static_root_serves_index_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "RAG Eval Harness" in resp.text


def test_api_health_not_shadowed_by_static_mount():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_legacy_health_still_works():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
