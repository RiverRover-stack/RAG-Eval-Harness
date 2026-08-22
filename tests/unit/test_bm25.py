from rag_eval.retrieval.bm25 import Bm25Index, Bm25Searcher, tokenize


def test_tokenizer_keeps_identifiers_with_underscores_whole():
    assert tokenize("Use response_model to filter output") == [
        "use",
        "response_model",
        "to",
        "filter",
        "output",
    ]


def test_tokenizer_keeps_identifiers_with_dots_whole():
    assert "jsonable_encoder" in tokenize("call jsonable_encoder(obj) to serialize")
    assert "fastapi.encoders" in tokenize("import fastapi.encoders")


def test_tokenizer_splits_on_ordinary_punctuation():
    assert tokenize("Query params, headers, and cookies!") == [
        "query",
        "params",
        "headers",
        "and",
        "cookies",
    ]


def _build_index(docs: dict[str, str]) -> Bm25Index:
    ids = list(docs.keys())
    documents = list(docs.values())
    metadatas = [{"url": f"https://x/{i}", "title": i} for i in ids]
    return Bm25Index.build(ids, documents, metadatas)


def test_planted_identifier_ranks_above_docs_without_it():
    index = _build_index(
        {
            "hit": "Use response_model to control the shape of the response body.",
            "miss-1": "FastAPI supports dependency injection out of the box.",
            "miss-2": "Path parameters are validated using type hints.",
        }
    )

    results = index.score_all("response_model", source_type="docs")
    results.sort(key=lambda c: c.final_score, reverse=True)

    assert results[0].chunk_id == "hit"
    assert all(c.final_score > 0 for c in results if c.chunk_id == "hit")


def test_score_all_omits_docs_with_no_query_terms():
    index = _build_index({"hit": "response_model controls serialization", "miss": "totally unrelated text"})
    results = index.score_all("response_model", source_type="docs")
    assert [c.chunk_id for c in results] == ["hit"]


def test_searcher_tags_source_type_from_indices_key_not_metadata():
    docs_index = _build_index({"d1": "response_model example"})
    disc_index = _build_index({"q1": "response_model example"})
    searcher = Bm25Searcher(indices={"docs": docs_index, "discussions": disc_index})

    results = searcher.search("response_model", k=5)

    source_types = {c.chunk_id: c.source_type for c in results}
    assert source_types == {"d1": "docs", "q1": "discussions"}


def test_searcher_excludes_deny_ids_and_assigns_rank():
    index = _build_index({"keep": "response_model here", "deny-me": "response_model there"})
    searcher = Bm25Searcher(indices={"docs": index})

    results = searcher.search("response_model", k=5, deny_ids={"deny-me"})

    assert [c.chunk_id for c in results] == ["keep"]
    assert results[0].ranks["bm25"] == 1
