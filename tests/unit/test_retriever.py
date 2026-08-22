"""rag/retriever.py::retrieve is a thin back-compat wrapper over
DenseSearcher (docs/plan.md Phase 6) -- the merge/rank/truncate logic
itself is tested against DenseSearcher directly in test_dense.py. These
tests only check the wrapper's wiring: it builds a DenseSearcher over both
collections and maps Candidate -> RetrievedChunk correctly."""

from unittest.mock import patch

from rag_eval.rag.retriever import retrieve
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE


def _hit(chunk_id: str, score: float, content: str = "text", url: str = "http://example.com") -> dict:
    return {"id": chunk_id, "content": content, "metadata": {"url": url}, "score": score}


def test_retrieve_queries_both_sources_and_maps_to_retrieved_chunk(fake_embedder):
    with patch(
        "rag_eval.rag.retriever.vector_query",
        return_value=[_hit("c1", 0.5, "content", "https://fastapi.tiangolo.com/tutorial/")],
    ) as mock_query:
        results = retrieve("question", k=1, embedder=fake_embedder)

    called_sources = {call.args[1] for call in mock_query.call_args_list}
    assert called_sources == {DISCUSSIONS_SOURCE, DOCS_SOURCE}
    assert len(results) == 1
    assert results[0].content == "content"
    assert results[0].source_id == "https://fastapi.tiangolo.com/tutorial/"
    assert results[0].score == 0.5


def test_retrieve_defaults_source_id_to_empty_string_when_url_missing(fake_embedder):
    hit = {"id": "c1", "content": "content", "metadata": {}, "score": 0.5}
    with patch("rag_eval.rag.retriever.vector_query", return_value=[hit]):
        results = retrieve("question", k=5, embedder=fake_embedder)

    assert results[0].source_id == ""
