import logging

from rag_eval.eval.gold import EvalItem, build_gold_index, resolve_all, resolve_gold_chunks

CHUNKS = [
    {
        "id": "chunk-1",
        "document": "...",
        "metadata": {"url": "https://fastapi.tiangolo.com/tutorial/query-params/#required"},
    },
    {
        "id": "chunk-2",
        "document": "...",
        "metadata": {"url": "https://fastapi.tiangolo.com/tutorial/query-params/#defaults"},
    },
    {
        "id": "chunk-3",
        "document": "...",
        # no explicit anchor -- url is the bare page, same as docs_chunker's
        # behaviour for a section with no {#id}
        "metadata": {"url": "https://fastapi.tiangolo.com/tutorial/body/"},
    },
]


def test_exact_anchor_match_resolves_at_anchor_granularity():
    index = build_gold_index(CHUNKS)
    item = EvalItem(
        id="1",
        dataset="d",
        question="q",
        gold_urls=["https://fastapi.tiangolo.com/tutorial/query-params/#required"],
    )
    resolved = resolve_gold_chunks(item, index)
    assert resolved.gold_chunk_ids == ["chunk-1"]
    assert resolved.gold_granularity == "anchor"


def test_bare_page_url_falls_back_to_every_chunk_on_the_page():
    index = build_gold_index(CHUNKS)
    item = EvalItem(
        id="2",
        dataset="d",
        question="q",
        gold_urls=["https://fastapi.tiangolo.com/tutorial/query-params/"],
    )
    resolved = resolve_gold_chunks(item, index)
    assert set(resolved.gold_chunk_ids) == {"chunk-1", "chunk-2"}
    assert resolved.gold_granularity == "page"


def test_bare_page_url_matching_a_no_anchor_chunk_stays_anchor_granularity():
    # chunk-3's own url has no #fragment (the section had no explicit
    # anchor id), so a gold url equal to that bare page is an *exact*
    # match, not a fallback.
    index = build_gold_index(CHUNKS)
    item = EvalItem(
        id="3", dataset="d", question="q", gold_urls=["https://fastapi.tiangolo.com/tutorial/body/"]
    )
    resolved = resolve_gold_chunks(item, index)
    assert resolved.gold_chunk_ids == ["chunk-3"]
    assert resolved.gold_granularity == "anchor"


def test_unresolvable_url_yields_no_chunks_and_warns(caplog):
    index = build_gold_index(CHUNKS)
    item = EvalItem(
        id="4",
        dataset="d",
        question="q",
        gold_urls=["https://fastapi.tiangolo.com/nowhere/#nope"],
    )
    with caplog.at_level(logging.WARNING):
        resolved = resolve_gold_chunks(item, index)
    assert resolved.gold_chunk_ids == []
    assert any("does not resolve" in r.message for r in caplog.records)


def test_multiple_gold_urls_union_and_page_flag_sticky():
    # one url resolves exactly, another needs the page fallback -- the item
    # as a whole is flagged "page" since that's the less trustworthy case.
    index = build_gold_index(CHUNKS)
    item = EvalItem(
        id="5",
        dataset="d",
        question="q",
        gold_urls=[
            "https://fastapi.tiangolo.com/tutorial/query-params/#required",
            "https://fastapi.tiangolo.com/tutorial/query-params/",
        ],
    )
    resolved = resolve_gold_chunks(item, index)
    assert set(resolved.gold_chunk_ids) == {"chunk-1", "chunk-2"}
    assert resolved.gold_granularity == "page"


def test_resolve_gold_chunks_does_not_mutate_input():
    index = build_gold_index(CHUNKS)
    item = EvalItem(
        id="6",
        dataset="d",
        question="q",
        gold_urls=["https://fastapi.tiangolo.com/tutorial/query-params/#required"],
    )
    resolve_gold_chunks(item, index)
    assert item.gold_chunk_ids == []
    assert item.gold_granularity == "anchor"


def test_resolve_all_processes_every_item():
    index = build_gold_index(CHUNKS)
    items = [
        EvalItem(
            id=str(i),
            dataset="d",
            question="q",
            gold_urls=["https://fastapi.tiangolo.com/tutorial/query-params/#required"],
        )
        for i in range(3)
    ]
    resolved = resolve_all(items, index)
    assert all(r.gold_chunk_ids == ["chunk-1"] for r in resolved)


def test_build_gold_index_skips_chunks_without_url():
    chunks = [{"id": "x", "document": "d", "metadata": {}}]
    index = build_gold_index(chunks)
    assert index.by_url == {}
    assert index.by_page == {}
