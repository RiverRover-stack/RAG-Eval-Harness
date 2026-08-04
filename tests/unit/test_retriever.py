from unittest.mock import patch

from rag_eval.rag.retriever import retrieve
from rag_eval.rag.vector_store import DISCUSSIONS_COLLECTION, DOCS_COLLECTION


def _hit(score: float, content: str = "text", url: str = "http://example.com") -> dict:
    return {"content": content, "metadata": {"url": url}, "score": score}


def test_retrieve_merges_both_collections_and_ranks_by_score():
    discussions_hits = [_hit(0.5, "from discussions"), _hit(0.9, "from discussions best")]
    docs_hits = [_hit(0.8, "from docs"), _hit(0.3, "from docs worst")]

    def fake_query(text, collection_name, k=5):
        return discussions_hits if collection_name == DISCUSSIONS_COLLECTION else docs_hits

    with patch("rag_eval.rag.retriever.vector_query", side_effect=fake_query):
        results = retrieve("question", k=4)

    assert [r.content for r in results] == [
        "from discussions best",
        "from docs",
        "from discussions",
        "from docs worst",
    ]
    assert [r.score for r in results] == [0.9, 0.8, 0.5, 0.3]


def test_retrieve_truncates_merged_results_to_k():
    discussions_hits = [_hit(0.9), _hit(0.8), _hit(0.7)]
    docs_hits = [_hit(0.95), _hit(0.6), _hit(0.5)]

    def fake_query(text, collection_name, k=5):
        return discussions_hits if collection_name == DISCUSSIONS_COLLECTION else docs_hits

    with patch("rag_eval.rag.retriever.vector_query", side_effect=fake_query):
        results = retrieve("question", k=2)

    assert len(results) == 2
    assert [r.score for r in results] == [0.95, 0.9]


def test_retrieve_queries_both_collections_with_requested_k():
    with patch("rag_eval.rag.retriever.vector_query", return_value=[]) as mock_query:
        retrieve("question", k=7)

    called_collections = {call.args[1] for call in mock_query.call_args_list}
    assert called_collections == {DISCUSSIONS_COLLECTION, DOCS_COLLECTION}
    assert all(call.kwargs["k"] == 7 for call in mock_query.call_args_list)


def test_retrieve_maps_metadata_url_to_source_id():
    with patch(
        "rag_eval.rag.retriever.vector_query",
        return_value=[_hit(0.5, "content", "https://fastapi.tiangolo.com/tutorial/")],
    ):
        results = retrieve("question", k=1)

    assert len(results) == 1
    assert results[0].source_id == "https://fastapi.tiangolo.com/tutorial/"


def test_retrieve_defaults_source_id_to_empty_string_when_url_missing():
    hit = {"content": "content", "metadata": {}, "score": 0.5}
    with patch("rag_eval.rag.retriever.vector_query", return_value=[hit]):
        results = retrieve("question", k=5)

    assert results[0].source_id == ""
