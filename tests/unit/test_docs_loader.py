import logging
from pathlib import Path

import pytest

from rag_eval.ingestion.docs_loader import (
    HeadingAnchor,
    clean_headings,
    convert_admonitions,
    iter_raw_docs,
    load_normalized_docs,
    normalize_doc,
    resolve_code_macros,
    strip_termy_tags,
)


# ---------------------------------------------------------------------------
# iter_raw_docs
# ---------------------------------------------------------------------------


def test_iter_raw_docs_yields_relative_paths_and_text(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    (docs_dir / "tutorial").mkdir(parents=True)
    (docs_dir / "index.md").write_text("# Home", encoding="utf-8")
    (docs_dir / "tutorial" / "first-steps.md").write_text("# First Steps", encoding="utf-8")

    results = list(iter_raw_docs(docs_dir))

    paths = [p.as_posix() for p, _ in results]
    assert paths == ["index.md", "tutorial/first-steps.md"]
    texts = dict((p.as_posix(), t) for p, t in results)
    assert texts["index.md"] == "# Home"


def test_iter_raw_docs_excludes_release_notes(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "release-notes.md").write_text("# Release notes", encoding="utf-8")
    (docs_dir / "index.md").write_text("# Home", encoding="utf-8")

    results = list(iter_raw_docs(docs_dir))

    assert [p.as_posix() for p, _ in results] == ["index.md"]


def test_iter_raw_docs_ignores_non_markdown_files(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "index.md").write_text("# Home", encoding="utf-8")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")

    results = list(iter_raw_docs(docs_dir))

    assert [p.as_posix() for p, _ in results] == ["index.md"]


# ---------------------------------------------------------------------------
# resolve_code_macros
# ---------------------------------------------------------------------------


@pytest.fixture
def docs_src_root(tmp_path: Path) -> Path:
    src_root = tmp_path / "docs_src"
    (src_root / "tutorial").mkdir(parents=True)
    (src_root / "tutorial" / "first_steps.py").write_text(
        "line1\nline2\nline3\nline4\n", encoding="utf-8"
    )
    return src_root


def test_resolve_code_macros_inlines_whole_file(docs_src_root: Path):
    text = "before\n{* ../../docs_src/tutorial/first_steps.py *}\nafter"

    result = resolve_code_macros(text, docs_src_root)

    assert result == "before\n```python\nline1\nline2\nline3\nline4\n```\nafter"


def test_resolve_code_macros_respects_ln_range(docs_src_root: Path):
    text = "{* ../../docs_src/tutorial/first_steps.py ln[1:2] *}"

    result = resolve_code_macros(text, docs_src_root)

    assert result == "```python\nline1\nline2\n```"


def test_resolve_code_macros_respects_ln_list_and_ranges(docs_src_root: Path):
    text = "{* ../../docs_src/tutorial/first_steps.py ln[1,3:4] *}"

    result = resolve_code_macros(text, docs_src_root)

    assert result == "```python\nline1\nline3\nline4\n```"


def test_resolve_code_macros_hl_attribute_does_not_restrict_lines(docs_src_root: Path):
    text = "{* ../../docs_src/tutorial/first_steps.py hl[2] *}"

    result = resolve_code_macros(text, docs_src_root)

    assert result == "```python\nline1\nline2\nline3\nline4\n```"


def test_resolve_code_macros_ignores_extra_leading_dotdot_segments(docs_src_root: Path):
    text = "{* ../../../docs_src/tutorial/first_steps.py *}"

    result = resolve_code_macros(text, docs_src_root)

    assert result == "```python\nline1\nline2\nline3\nline4\n```"


def test_resolve_code_macros_missing_file_emits_placeholder_and_warns(
    docs_src_root: Path, caplog: pytest.LogCaptureFixture
):
    text = "{* ../../docs_src/tutorial/missing.py *}"

    with caplog.at_level(logging.WARNING):
        result = resolve_code_macros(text, docs_src_root)

    assert result == "```python\n# snippet not found: ../../docs_src/tutorial/missing.py\n```"
    assert "snippet file not found" in caplog.text


# ---------------------------------------------------------------------------
# convert_admonitions
# ---------------------------------------------------------------------------


def test_convert_admonitions_basic_tip():
    text = "/// tip\nThis is a tip.\n///"

    result = convert_admonitions(text)

    assert result == "*Tip*\nThis is a tip."


def test_convert_admonitions_with_title():
    text = "/// note | Custom Title\nBody text.\n///"

    result = convert_admonitions(text)

    assert result == "*Note: Custom Title*\nBody text."


def test_convert_admonitions_nested_tabs():
    text = "/// tip\nbefore\n//// tab | A\ninside A\n////\nafter\n///"

    result = convert_admonitions(text)

    assert result == "*Tip*\nbefore\n*Tab: A*\ninside A\nafter"


def test_convert_admonitions_leaves_plain_text_untouched():
    text = "Just a regular paragraph.\nNo admonitions here."

    assert convert_admonitions(text) == text


# ---------------------------------------------------------------------------
# strip_termy_tags
# ---------------------------------------------------------------------------


def test_strip_termy_tags_removes_wrapper_and_inline_tags():
    text = (
        '<div class="termy">\n'
        '$ <font color="red">uvicorn</font> main:app\n'
        "<span>INFO</span>: done\n"
        "</div>"
    )

    result = strip_termy_tags(text)

    assert result == "$ uvicorn main:app\nINFO: done"


def test_strip_termy_tags_unescapes_html_entities():
    text = '<div class="termy">\n$ echo &amp; done\n</div>'

    result = strip_termy_tags(text)

    assert result == "$ echo & done"


def test_strip_termy_tags_leaves_non_termy_html_alone():
    text = 'before\n<div class="note">not termy</div>\nafter'

    assert strip_termy_tags(text) == text


# ---------------------------------------------------------------------------
# clean_headings
# ---------------------------------------------------------------------------


def test_clean_headings_strips_anchor_suffix_and_returns_anchors():
    text = "## Query Parameters { #query-params }\nSome text\n### Sub { #sub-anchor }\nMore"

    cleaned, anchors = clean_headings(text)

    assert cleaned == "## Query Parameters\nSome text\n### Sub\nMore"
    assert anchors == [
        HeadingAnchor(level=2, title="Query Parameters", anchor="query-params"),
        HeadingAnchor(level=3, title="Sub", anchor="sub-anchor"),
    ]


def test_clean_headings_leaves_headings_without_anchor_untouched():
    text = "## No Anchor Heading\ntext"

    cleaned, anchors = clean_headings(text)

    assert cleaned == text
    assert anchors == []


# ---------------------------------------------------------------------------
# normalize_doc / load_normalized_docs (pipeline)
# ---------------------------------------------------------------------------


def test_normalize_doc_runs_all_steps(docs_src_root: Path):
    raw_text = (
        "# Title { #title }\n"
        "/// tip\n"
        "See below.\n"
        "///\n"
        "{* ../../docs_src/tutorial/first_steps.py ln[1:1] *}\n"
    )

    doc = normalize_doc(Path("index.md"), raw_text, docs_src_root)

    assert doc.path == Path("index.md")
    assert doc.raw_text == raw_text
    assert "```python\nline1\n```" in doc.text
    assert "*Tip*" in doc.text
    assert "{ #title }" not in doc.text
    assert doc.anchors == [HeadingAnchor(level=1, title="Title", anchor="title")]


def test_load_normalized_docs_end_to_end(tmp_path: Path, docs_src_root: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "index.md").write_text("# Home { #home }\nWelcome.", encoding="utf-8")

    docs = list(load_normalized_docs(docs_dir, docs_src_root))

    assert len(docs) == 1
    assert docs[0].path == Path("index.md")
    assert docs[0].anchors == [HeadingAnchor(level=1, title="Home", anchor="home")]
