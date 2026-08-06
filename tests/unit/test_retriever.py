from unittest.mock import patch

from rag_eval.rag.retriever import retrieve
from rag_eval.rag.vector_store import DISCUSSIONS_SOURCE, DOCS_SOURCE


def _hit(score: float, content: str = "text", url: str = "http://example.com") -> dict:
    return {"content": content, "metadata": {"url": url}, "score": score}


def test_retrieve_merges_both_sources_and_ranks_by_score(fake_embedder):
    discussions_hits = [_hit(0.5, "from discussions"), _hit(0.9, "from discussions best")]
    docs_hits = [_hit(0.8, "from docs"), _hit(0.3, "from docs worst")]

    def fake_query(embedding, source, embedder, k=5):
        return discussions_hits if source == DISCUSSIONS_SOURCE else docs_hits

    with patch("rag_eval.rag.retriever.vector_query", side_effect=fake_query):
        results = retrieve("question", k=4, embedder=fake_embedder)

    assert [r.content for r in results] == [
        "from discussions best",
        "from docs",
        "from discussions",
        "from docs worst",
    ]
    assert [r.score for r in results] == [0.9, 0.8, 0.5, 0.3]


def test_retrieve_truncates_merged_results_to_k(fake_embedder):
    discussions_hits = [_hit(0.9), _hit(0.8), _hit(0.7)]
    docs_hits = [_hit(0.95), _hit(0.6), _hit(0.5)]

    def fake_query(embedding, source, embedder, k=5):
        return discussions_hits if source == DISCUSSIONS_SOURCE else docs_hits

    with patch("rag_eval.rag.retriever.vector_query", side_effect=fake_query):
        results = retrieve("question", k=2, embedder=fake_embedder)

    assert len(results) == 2
    assert [r.score for r in results] == [0.95, 0.9]


def test_retrieve_queries_both_sources_with_requested_k(fake_embedder):
    with patch("rag_eval.rag.retriever.vector_query", return_value=[]) as mock_query:
        retrieve("question", k=7, embedder=fake_embedder)

    called_sources = {call.args[1] for call in mock_query.call_args_list}
    assert called_sources == {DISCUSSIONS_SOURCE, DOCS_SOURCE}
    assert all(call.kwargs["k"] == 7 for call in mock_query.call_args_list)
    assert all(call.args[2] is fake_embedder for call in mock_query.call_args_list)


def test_retrieve_maps_metadata_url_to_source_id(fake_embedder):
    with patch(
        "rag_eval.rag.retriever.vector_query",
        return_value=[_hit(0.5, "content", "https://fastapi.tiangolo.com/tutorial/")],
    ):
        results = retrieve("question", k=1, embedder=fake_embedder)

    assert len(results) == 1
    assert results[0].source_id == "https://fastapi.tiangolo.com/tutorial/"


def test_retrieve_defaults_source_id_to_empty_string_when_url_missing(fake_embedder):
    hit = {"content": "content", "metadata": {}, "score": 0.5}
    with patch("rag_eval.rag.retriever.vector_query", return_value=[hit]):
        results = retrieve("question", k=5, embedder=fake_embedder)

    assert results[0].source_id == ""
