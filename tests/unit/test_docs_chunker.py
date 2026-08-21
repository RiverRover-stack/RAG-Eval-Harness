import logging
from pathlib import Path

import pytest

from rag_eval.ingestion.docs_chunker import (
    CHARS_PER_TOKEN,
    TARGET_MAX_TOKENS,
    _assign_anchors,
    _atomic_blocks,
    _doc_url,
    _estimate_tokens,
    _merge_undersized,
    _pack_blocks,
    _split_into_blocks,
    _split_oversized_block,
    _split_sections,
    doc_to_chunks,
    load_doc_chunks,
)
from rag_eval.ingestion.docs_loader import HeadingAnchor, NormalizedDoc


def _block(n_tokens: int) -> str:
    """A text blob that _estimate_tokens sizes at exactly n_tokens."""
    return "a" * (n_tokens * CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# _split_sections
# ---------------------------------------------------------------------------


def test_split_sections_splits_on_h2_and_h3_and_tracks_breadcrumbs():
    text = (
        "Intro text.\n\n"
        "## Section A\n"
        "Content A\n\n"
        "### Sub A1\n"
        "Content A1\n\n"
        "## Section B\n"
        "Content B"
    )

    sections = _split_sections(text)

    assert [s.breadcrumb for s in sections] == ["", "Section A", "Section A > Sub A1", "Section B"]
    assert [s.heading_title for s in sections] == [None, "Section A", "Sub A1", "Section B"]


def test_split_sections_h3_before_any_h2_is_its_own_top_level_breadcrumb():
    text = "### Standalone Sub\nContent"

    sections = _split_sections(text)

    assert len(sections) == 1
    assert sections[0].breadcrumb == "Standalone Sub"


def test_split_sections_ignores_hash_comment_inside_fenced_code():
    text = "## Section\n```python\n# This is not asynchronous\nx = 1\n```\nEnd"

    sections = _split_sections(text)

    assert len(sections) == 1
    assert sections[0].breadcrumb == "Section"
    assert "# This is not asynchronous" in sections[0].lines


def test_split_sections_h4_and_deeper_stay_folded_into_parent_section():
    text = "## Section\n#### Sub-sub\nContent"

    sections = _split_sections(text)

    assert len(sections) == 1
    assert "#### Sub-sub" in sections[0].lines


def test_split_sections_drops_leading_section_when_no_content_before_first_heading():
    text = "## Section\nContent"

    sections = _split_sections(text)

    assert len(sections) == 1
    assert sections[0].breadcrumb == "Section"


# ---------------------------------------------------------------------------
# _assign_anchors
# ---------------------------------------------------------------------------


def test_assign_anchors_pairs_sections_with_anchors_in_order():
    text = "## Section A\nContent A\n\n## Section B\nContent B"
    sections = _split_sections(text)
    anchors = [
        HeadingAnchor(level=2, title="Section A", anchor="section-a"),
        HeadingAnchor(level=2, title="Section B", anchor="section-b"),
    ]

    _assign_anchors(sections, anchors, "page.md")

    assert [s.anchor for s in sections] == ["section-a", "section-b"]


def test_assign_anchors_leaves_missing_anchor_as_none_and_warns(caplog: pytest.LogCaptureFixture):
    text = "## Section A\nContent A\n\n## Section B\nContent B"
    sections = _split_sections(text)
    anchors = [HeadingAnchor(level=2, title="Section A", anchor="section-a")]

    with caplog.at_level(logging.WARNING):
        _assign_anchors(sections, anchors, "page.md")

    assert [s.anchor for s in sections] == ["section-a", None]
    assert "page.md" in caplog.text


def test_assign_anchors_skips_section_with_no_heading_title():
    text = "Intro\n\n## Section A\nContent A"
    sections = _split_sections(text)
    anchors = [HeadingAnchor(level=2, title="Section A", anchor="section-a")]

    _assign_anchors(sections, anchors, "page.md")

    assert sections[0].heading_title is None
    assert sections[0].anchor is None
    assert sections[1].anchor == "section-a"


# ---------------------------------------------------------------------------
# _split_into_blocks
# ---------------------------------------------------------------------------


def test_split_into_blocks_separates_prose_and_code():
    lines = [
        "Para one line1.",
        "Para one line2.",
        "",
        "```py",
        "code line",
        "```",
        "",
        "Para two.",
    ]

    blocks = _split_into_blocks(lines)

    assert blocks == [
        ("Para one line1.\nPara one line2.", False),
        ("```py\ncode line\n```", True),
        ("Para two.", False),
    ]


def test_split_into_blocks_treats_blank_line_inside_fence_as_code_content():
    lines = ["```py", "line1", "", "line2", "```"]

    blocks = _split_into_blocks(lines)

    assert blocks == [("```py\nline1\n\nline2\n```", True)]


# ---------------------------------------------------------------------------
# _split_oversized_block
# ---------------------------------------------------------------------------


def test_split_oversized_block_breaks_at_max_token_boundary():
    lines = [_block(100)] * 7
    block = "\n".join(lines)

    parts = _split_oversized_block(block)

    assert len(parts) == 2
    assert parts[0] == "\n".join(lines[:6])
    assert parts[1] == lines[6]
    # split point is driven by the running sum of per-line token estimates,
    # not the joined string's own estimate (which also counts the newlines)
    assert sum(_estimate_tokens(l) for l in lines[:6]) == TARGET_MAX_TOKENS


def test_split_oversized_block_single_short_line_stays_one_part():
    parts = _split_oversized_block(_block(10))

    assert parts == [_block(10)]


# ---------------------------------------------------------------------------
# _atomic_blocks
# ---------------------------------------------------------------------------


def test_atomic_blocks_never_splits_a_fenced_code_block_even_when_oversized():
    code_lines = [_block(100)] * 7
    lines = ["```python", *code_lines, "```"]

    blocks = _atomic_blocks(lines)

    assert len(blocks) == 1
    assert blocks[0] == "\n".join(lines)
    assert _estimate_tokens(blocks[0]) > TARGET_MAX_TOKENS


def test_atomic_blocks_splits_oversized_prose_block():
    lines = [_block(100)] * 7  # one prose block, no blank lines inside

    blocks = _atomic_blocks(lines)

    assert len(blocks) > 1
    assert sum(b.count("\n") + 1 for b in blocks) == len(lines)  # no lines lost or duplicated
    # per-block token sum (the split's own accounting) stays within the cap; the joined
    # string's own estimate can run a little higher since it also counts the newlines
    assert all(_estimate_tokens(b) <= TARGET_MAX_TOKENS + len(b.splitlines()) for b in blocks)


def test_atomic_blocks_leaves_small_blocks_untouched():
    lines = ["Short paragraph.", "", "Another short one."]

    blocks = _atomic_blocks(lines)

    assert blocks == ["Short paragraph.", "Another short one."]


# ---------------------------------------------------------------------------
# _pack_blocks
# ---------------------------------------------------------------------------


def test_pack_blocks_keeps_absorbing_below_floor_even_past_ceiling():
    blocks = [_block(300), _block(350)]

    packed = _pack_blocks(blocks)

    assert len(packed) == 1
    assert packed[0] == blocks[0] + "\n\n" + blocks[1]


def test_pack_blocks_splits_once_floor_is_reached_and_next_block_overflows():
    blocks = [_block(420), _block(300)]

    packed = _pack_blocks(blocks)

    assert packed == [blocks[0], blocks[1]]


def test_pack_blocks_merges_when_staying_within_ceiling():
    blocks = [_block(300), _block(300)]

    packed = _pack_blocks(blocks)

    assert len(packed) == 1


def test_pack_blocks_oversized_single_block_forms_its_own_chunk():
    blocks = [_block(700)]

    packed = _pack_blocks(blocks)

    assert packed == [blocks[0]]


# ---------------------------------------------------------------------------
# _merge_undersized
# ---------------------------------------------------------------------------


def _raw_chunk(section: str, n_tokens: int, url: str = "") -> dict:
    return {"document": _block(n_tokens), "metadata": {"section": section, "url": url}}


def test_merge_undersized_merges_consecutive_chunks_sharing_h2_parent():
    chunks = [
        _raw_chunk("Parent", 30, url="https://example.com/page/#parent"),
        _raw_chunk("Parent > Child A", 30, url="https://example.com/page/#child-a"),
        _raw_chunk("Parent > Child B", 30, url="https://example.com/page/#child-b"),
    ]

    merged = _merge_undersized(chunks, min_tokens=150, drop_below=25)

    assert len(merged) == 1
    assert merged[0]["document"] == "\n\n".join(c["document"] for c in chunks)
    assert merged[0]["metadata"]["section"] == "Parent"  # keeps the first chunk's metadata
    assert merged[0]["metadata"]["url"] == "https://example.com/page/#parent"
    # the swallowed subsections' urls must stay resolvable for gold labels
    assert merged[0]["metadata"]["merged_urls"] == (
        "https://example.com/page/#child-a|https://example.com/page/#child-b"
    )


def test_merge_undersized_stops_absorbing_once_floor_is_reached():
    chunks = [_raw_chunk("Parent", 160), _raw_chunk("Parent > Child A", 30)]

    merged = _merge_undersized(chunks, min_tokens=150, drop_below=25)

    assert len(merged) == 2


def test_merge_undersized_does_not_merge_across_different_h2_parents():
    chunks = [_raw_chunk("Section One", 30), _raw_chunk("Section Two", 30)]

    merged = _merge_undersized(chunks, min_tokens=150, drop_below=25)

    assert len(merged) == 2


def test_merge_undersized_drops_chunks_still_under_the_floor():
    chunks = [_raw_chunk("Section One", 30), _raw_chunk("Section Two", 10)]

    merged = _merge_undersized(chunks, min_tokens=150, drop_below=25)

    assert len(merged) == 1
    assert merged[0]["metadata"]["section"] == "Section One"


def test_doc_to_chunks_merges_short_h3_siblings_under_one_h2_parent():
    text = (
        f"## Parent\n{_block(10)}\n\n"
        f"### Child A\n{_block(10)}\n\n"
        f"### Child B\n{_block(10)}"
    )
    anchors = [
        HeadingAnchor(level=2, title="Parent", anchor="parent"),
        HeadingAnchor(level=3, title="Child A", anchor="child-a"),
        HeadingAnchor(level=3, title="Child B", anchor="child-b"),
    ]
    doc = NormalizedDoc(path=Path("guide.md"), raw_text=text, text=text, anchors=anchors)

    chunks = doc_to_chunks(doc, base_url="https://example.com")

    assert len(chunks) == 1
    assert len({c["metadata"]["parent_id"] for c in chunks}) == 1
    # Child A's and Child B's urls must still resolve to this merged chunk
    assert chunks[0]["metadata"]["url"] == "https://example.com/guide/#parent"
    assert chunks[0]["metadata"]["merged_urls"] == (
        "https://example.com/guide/#child-a|https://example.com/guide/#child-b"
    )


def test_doc_to_chunks_gives_different_h2_parents_different_parent_ids():
    text = f"## Section One\n{_block(200)}\n\n## Section Two\n{_block(200)}"
    anchors = [
        HeadingAnchor(level=2, title="Section One", anchor="section-one"),
        HeadingAnchor(level=2, title="Section Two", anchor="section-two"),
    ]
    doc = NormalizedDoc(path=Path("guide.md"), raw_text=text, text=text, anchors=anchors)

    chunks = doc_to_chunks(doc, base_url="https://example.com")

    assert len(chunks) == 2
    assert len({c["metadata"]["parent_id"] for c in chunks}) == 2


# ---------------------------------------------------------------------------
# _doc_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "anchor", "expected"),
    [
        ("index.md", None, "https://example.com/"),
        ("tutorial/index.md", None, "https://example.com/tutorial/"),
        ("tutorial/query-params.md", None, "https://example.com/tutorial/query-params/"),
        ("tutorial/query-params.md", "defaults", "https://example.com/tutorial/query-params/#defaults"),
    ],
)
def test_doc_url(path, anchor, expected):
    assert _doc_url(path, anchor, "https://example.com") == expected


def test_doc_url_strips_trailing_slash_on_base_url():
    assert _doc_url("index.md", None, "https://example.com/") == "https://example.com/"


# ---------------------------------------------------------------------------
# doc_to_chunks / load_doc_chunks (pipeline)
# ---------------------------------------------------------------------------


def test_doc_to_chunks_produces_one_chunk_per_section_with_metadata():
    # Each section's content is >=25 tokens (the drop floor) and the three
    # sections are siblings (different ##/no-heading top levels), so none
    # of _merge_undersized's merge-or-drop pass fires here.
    text = (
        "# Page Title\n"
        f"Intro paragraph. {_block(30)}\n\n"
        "## Section One\n"
        f"Content of section one. {_block(30)}\n\n"
        "## Section Two\n"
        f"Content of section two. {_block(30)}"
    )
    anchors = [
        HeadingAnchor(level=1, title="Page Title", anchor="page-title"),
        HeadingAnchor(level=2, title="Section One", anchor="section-one"),
        HeadingAnchor(level=2, title="Section Two", anchor="section-two"),
    ]
    doc = NormalizedDoc(path=Path("tutorial/example.md"), raw_text=text, text=text, anchors=anchors)

    chunks = doc_to_chunks(doc, base_url="https://example.com")

    assert len(chunks) == 3
    sections = [c["metadata"]["section"] for c in chunks]
    assert sections == ["", "Section One", "Section Two"]
    assert all(c["metadata"]["title"] == "Page Title" for c in chunks)
    assert all(c["metadata"]["source_type"] == "docs" for c in chunks)
    assert all(c["metadata"]["path"] == "tutorial/example.md" for c in chunks)
    assert chunks[0]["metadata"]["url"] == "https://example.com/tutorial/example/"
    assert chunks[1]["metadata"]["url"] == "https://example.com/tutorial/example/#section-one"
    assert chunks[2]["metadata"]["url"] == "https://example.com/tutorial/example/#section-two"
    assert {c["id"] for c in chunks} == {c["id"] for c in chunks}  # all present
    assert len({c["id"] for c in chunks}) == 3  # and unique
    assert all(len(c["metadata"]["content_hash"]) == 16 for c in chunks)
    # three distinct ## parents (no heading, Section One, Section Two) -> three distinct parent_ids
    assert len({c["metadata"]["parent_id"] for c in chunks}) == 3


def test_doc_to_chunks_chunk_index_is_per_section():
    text = "## Section\n" + "\n\n".join(_block(300) for _ in range(4))
    doc = NormalizedDoc(
        path=Path("big.md"),
        raw_text=text,
        text=text,
        anchors=[HeadingAnchor(level=2, title="Section", anchor="section")],
    )

    chunks = doc_to_chunks(doc, base_url="https://example.com")

    assert len(chunks) > 1
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_load_doc_chunks_end_to_end(tmp_path: Path):
    # Section bodies need to clear the 25-token drop floor (_merge_undersized)
    # to survive into the output, since these two sections are siblings
    # (different top-level ##s) and so are never merged with each other.
    docs_dir = tmp_path / "docs"
    docs_src_dir = tmp_path / "docs_src"
    docs_dir.mkdir()
    docs_src_dir.mkdir()
    (docs_dir / "index.md").write_text(
        f"# Home {{ #home }}\n\nWelcome to the docs. {_block(30)}\n\n"
        f"## Getting Started {{ #getting-started }}\n\nRun `pip install foo`. {_block(30)}",
        encoding="utf-8",
    )

    chunks = list(load_doc_chunks(docs_dir, docs_src_dir, base_url="https://example.com"))

    assert len(chunks) == 2
    assert {c["metadata"]["section"] for c in chunks} == {"", "Getting Started"}
    assert all(c["metadata"]["path"] == "index.md" for c in chunks)
