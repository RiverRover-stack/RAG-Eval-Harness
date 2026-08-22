"""DenseSearcher owns the cross-collection merge that used to live directly
in rag/retriever.py::retrieve (docs/plan.md Phase 6). query_fn is injected
so these never touch a real Chroma collection or embedder (CLAUDE.md:
prefer constructor injection over patching)."""

from rag_eval.retrieval.dense import DenseSearcher

SOURCES = ["discussions", "docs"]


def _hit(chunk_id: str, score: float, content: str = "text", url: str = "http://example.com") -> dict:
    return {"id": chunk_id, "content": content, "metadata": {"url": url}, "score": score}


def test_search_merges_both_sources_and_ranks_by_score(fake_embedder):
    discussions_hits = [_hit("d1", 0.5, "from discussions"), _hit("d2", 0.9, "from discussions best")]
    docs_hits = [_hit("c1", 0.8, "from docs"), _hit("c2", 0.3, "from docs worst")]

    def fake_query(embedding, source, embedder, k=5):
        return discussions_hits if source == "discussions" else docs_hits

    searcher = DenseSearcher(sources=SOURCES, embedder=fake_embedder, query_fn=fake_query)
    results = searcher.search("question", k=4)

    assert [c.content for c in results] == [
        "from discussions best",
        "from docs",
        "from discussions",
        "from docs worst",
    ]
    assert [c.final_score for c in results] == [0.9, 0.8, 0.5, 0.3]


def test_search_truncates_merged_results_to_k(fake_embedder):
    discussions_hits = [_hit("d1", 0.9), _hit("d2", 0.8), _hit("d3", 0.7)]
    docs_hits = [_hit("c1", 0.95), _hit("c2", 0.6), _hit("c3", 0.5)]

    def fake_query(embedding, source, embedder, k=5):
        return discussions_hits if source == "discussions" else docs_hits

    searcher = DenseSearcher(sources=SOURCES, embedder=fake_embedder, query_fn=fake_query)
    results = searcher.search("question", k=2)

    assert len(results) == 2
    assert [c.final_score for c in results] == [0.95, 0.9]


def test_search_queries_every_source_with_the_over_fetch_k(fake_embedder):
    calls = []

    def fake_query(embedding, source, embedder, k=5):
        calls.append((source, embedder, k))
        return []

    searcher = DenseSearcher(sources=SOURCES, embedder=fake_embedder, query_fn=fake_query)
    searcher.search("question", k=7)

    called_sources = {source for source, _, _ in calls}
    assert called_sources == {"discussions", "docs"}
    assert all(k == 7 for _, _, k in calls)
    assert all(embedder is fake_embedder for _, embedder, _ in calls)


def test_search_maps_metadata_url_to_candidate_url(fake_embedder):
    hit = _hit("c1", 0.5, "content", "https://fastapi.tiangolo.com/tutorial/")
    searcher = DenseSearcher(sources=SOURCES, embedder=fake_embedder, query_fn=lambda *a, **k: [hit])

    results = searcher.search("question", k=1)

    assert len(results) == 1
    assert results[0].url == "https://fastapi.tiangolo.com/tutorial/"


def test_search_defaults_url_to_empty_string_when_missing(fake_embedder):
    hit = {"id": "c1", "content": "content", "metadata": {}, "score": 0.5}
    searcher = DenseSearcher(sources=SOURCES, embedder=fake_embedder, query_fn=lambda *a, **k: [hit])

    results = searcher.search("question", k=5)

    assert results[0].url == ""


def test_search_excludes_deny_ids(fake_embedder):
    def fake_query(embedding, source, embedder, k=5):
        return [_hit("keep", 0.9)] if source == "discussions" else [_hit("deny-me", 0.8)]

    searcher = DenseSearcher(sources=SOURCES, embedder=fake_embedder, query_fn=fake_query)
    results = searcher.search("question", k=5, deny_ids={"deny-me"})

    assert [c.chunk_id for c in results] == ["keep"]
