"""
Turn normalized FastAPI doc pages (see docs_loader.py) into embeddable
chunks, structurally split on `##`/`###` headings.

Metadata shape is kept consistent with the discussions chunker
(chunker.py): source_type, title, url, chunk_index, parent_id, content_hash
are common to every chunk regardless of source; section/path are docs-only.

Chunking strategy:
  1. Split each page into sections at `##`/`###` headings (never at `#`,
     the page's own title, or `####`+, which stay folded into their
     nearest `##`/`###` ancestor's section).
  2. Within a section, split into atomic blocks at blank lines, treating
     each fenced ```code``` block as a single atomic block that is never
     split (packing.py).
  3. Greedily pack those blocks into chunks targeting 400-600 tokens
     (approximated at ~4 chars/token -- there's no exact tokenizer here
     since embeddings come from a local Ollama model, not an OpenAI-style
     tokenizer). The 400 floor is prioritized over the 600 ceiling: a
     chunk keeps absorbing blocks until it has *reached* 400 tokens, even
     if that means overshooting 600 for one chunk. The only hard
     invariant is that a fenced code block is never split to hit either
     bound (packing.py).
  4. Since packing happens per-section, a page with many short `###`
     subsections under one `##` parent still emits one micro-chunk per
     subsection. A post-pass (`_merge_undersized`) merges consecutive
     chunks that share a `##` ancestor and are still under 150 tokens,
     then drops whatever's left under 25 tokens.

Usage:
    uv run python -m rag_eval.ingestion.docs_chunker
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag_eval.common.config import settings
from rag_eval.ingestion.docs_loader import (
    DOCS_DIR,
    DOCS_SRC_DIR,
    HeadingAnchor,
    NormalizedDoc,
    load_normalized_docs,
)
from rag_eval.ingestion.packing import (
    CHARS_PER_TOKEN,
    TARGET_MAX_TOKENS,
    _atomic_blocks,
    _estimate_tokens,
    _pack_blocks,
    _split_into_blocks,
    _split_oversized_block,
)

logger = logging.getLogger(__name__)

# Re-exported from packing.py so existing imports of these names from
# docs_chunker keep working now that they live in a shared module.
__all__ = [
    "CHARS_PER_TOKEN",
    "TARGET_MAX_TOKENS",
    "_atomic_blocks",
    "_estimate_tokens",
    "_pack_blocks",
    "_split_into_blocks",
    "_split_oversized_block",
    "doc_to_chunks",
    "load_doc_chunks",
]

# Undersized-chunk merge pass (see doc_to_chunks / _merge_undersized): looser
# than packing.py's own 400-token floor, since this runs across whole
# sections rather than within one.
MERGE_MIN_TOKENS = 150
DROP_BELOW_TOKENS = 25


def _content_hash(text: str) -> str:
    """Hash of a chunk's embedded text, used to skip re-embedding unchanged
    chunks on re-ingestion."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Split into ##/### sections
# ---------------------------------------------------------------------------

_FENCE_PREFIX = "```"


@dataclass
class _Section:
    breadcrumb: str
    heading_title: str | None
    anchor: str | None = None
    lines: list[str] = field(default_factory=list)


def _parse_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes < 1 or hashes > 6:
        return None
    rest = stripped[hashes:]
    if not rest[:1].isspace():
        return None
    return hashes, rest.strip()


def _split_sections(text: str) -> list[_Section]:
    """Split normalized doc text into sections at `##`/`###` headings,
    tracking fenced-code state so a `#`-style comment inside a code block
    (e.g. `# This is not asynchronous`) is never mistaken for a heading."""
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current = _Section(breadcrumb="", heading_title=None)
    in_fence = False

    for line in text.splitlines():
        if line.strip().startswith(_FENCE_PREFIX):
            in_fence = not in_fence
            current.lines.append(line)
            continue

        heading = None if in_fence else _parse_heading(line)
        if heading and heading[0] in (2, 3):
            sections.append(current)
            level, title = heading
            if level == 2:
                heading_stack = [(2, title)]
            elif heading_stack and heading_stack[0][0] == 2:
                heading_stack = heading_stack[:1] + [(3, title)]
            else:
                heading_stack = [(3, title)]
            breadcrumb = " > ".join(t for _, t in heading_stack)
            current = _Section(breadcrumb=breadcrumb, heading_title=title, lines=[line])
            continue

        current.lines.append(line)

    sections.append(current)
    return [s for s in sections if any(line.strip() for line in s.lines)]


def _assign_anchors(sections: list[_Section], anchors: list[HeadingAnchor], path_str: str) -> None:
    """Pair each ##/### section with its heading's anchor id (from
    docs_loader.clean_headings), in document order.

    Some pages (e.g. release-notes.md) have headings with no explicit
    `{ #id }` -- mkdocs-material would still auto-slugify an anchor for
    them, but docs_loader.clean_headings only captures explicit ids, so
    those sections are left with anchor=None and their chunk url falls
    back to the bare page URL instead of a deep link.
    """
    split_anchors = [a for a in anchors if a.level in (2, 3)]
    idx = 0
    missing = 0
    for section in sections:
        if section.heading_title is None:
            continue
        if idx >= len(split_anchors):
            missing += 1
            continue
        section.anchor = split_anchors[idx].anchor
        idx += 1
    if missing:
        logger.warning(
            "docs_chunker: %s has %d ##/### section(s) with no explicit anchor id; "
            "their chunk urls have no #fragment",
            path_str,
            missing,
        )


# ---------------------------------------------------------------------------
# 2. Undersized-chunk merge pass (post per-section packing)
# ---------------------------------------------------------------------------


def _h2_breadcrumb(breadcrumb: str) -> str:
    """The top-level `##` ancestor of a section's breadcrumb -- used to
    group `###` siblings for both the merge pass below and parent_id."""
    return breadcrumb.split(" > ")[0]


def _merge_undersized(
    chunks: list[dict], min_tokens: int = MERGE_MIN_TOKENS, drop_below: int = DROP_BELOW_TOKENS
) -> list[dict]:
    """Merge consecutive chunks that share a `##` ancestor and are still
    under `min_tokens`, then drop whatever's left under `drop_below`.

    `_pack_blocks` packs only within one section, so a page with many short
    `###` subsections under one `##` parent still emits one micro-chunk per
    subsection (avg 210 tokens, 134 under 50 across the corpus before this
    pass). Chunks are assumed to already be in document order.

    A merged chunk keeps the *first* absorbed subsection's `url` as its
    primary metadata (used for citations), but every other subsection's url
    it swallows is recorded in `merged_urls` (`"|"`-joined -- Chroma
    metadata values must be scalar, not a list) so gold-label resolution can
    still find this chunk when a gold URL names one of the swallowed
    subsections instead of the first one.
    """
    merged: list[dict] = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and _estimate_tokens(prev["document"]) < min_tokens
            and _h2_breadcrumb(prev["metadata"]["section"])
            == _h2_breadcrumb(chunk["metadata"]["section"])
        ):
            prev["document"] = prev["document"] + "\n\n" + chunk["document"]
            absorbed_url = chunk["metadata"].get("url", "")
            if absorbed_url and absorbed_url != prev["metadata"].get("url", ""):
                urls = prev["metadata"]["merged_urls"].split("|") if prev["metadata"]["merged_urls"] else []
                if absorbed_url not in urls:
                    urls.append(absorbed_url)
                prev["metadata"]["merged_urls"] = "|".join(urls)
        else:
            merged.append(
                {
                    "document": chunk["document"],
                    "metadata": {**chunk["metadata"], "merged_urls": ""},
                }
            )

    return [c for c in merged if _estimate_tokens(c["document"]) >= drop_below]


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


def _doc_url(path: str, anchor: str | None, base_url: str) -> str:
    """Live FastAPI docs URL for a path + anchor, e.g.
    `tutorial/query-params.md` (+ `#defaults`) ->
    `https://fastapi.tiangolo.com/tutorial/query-params/#defaults`."""
    stem = path.removesuffix(".md")
    if stem == "index":
        url_path = ""
    elif stem.endswith("/index"):
        url_path = stem[: -len("index")]
    else:
        url_path = f"{stem}/"

    url = f"{base_url.rstrip('/')}/{url_path}"
    if anchor:
        url = f"{url}#{anchor}"
    return url


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def doc_to_chunks(doc: NormalizedDoc, base_url: str = settings.docs_base_url) -> list[dict]:
    """Chunk one normalized doc page into Chroma-ready {id, document, metadata} dicts."""
    path_str = doc.path.as_posix()
    page_title = next((a.title for a in doc.anchors if a.level == 1), "")

    sections = _split_sections(doc.text)
    _assign_anchors(sections, doc.anchors, path_str)

    raw_chunks: list[dict] = []
    for section in sections:
        blocks = _atomic_blocks(section.lines)
        if not blocks:
            continue
        for chunk_text in _pack_blocks(blocks):
            raw_chunks.append(
                {
                    "document": chunk_text,
                    "metadata": {
                        "source_type": "docs",
                        "title": page_title,
                        "section": section.breadcrumb,
                        "path": path_str,
                        "url": _doc_url(path_str, section.anchor, base_url),
                    },
                }
            )

    chunks: list[dict] = []
    for chunk_index, chunk in enumerate(_merge_undersized(raw_chunks)):
        meta = chunk["metadata"]
        document = chunk["document"]
        chunks.append(
            {
                "id": _content_hash(f"{path_str}::{meta['section']}::{chunk_index}"),
                "document": document,
                "metadata": {
                    **meta,
                    "chunk_index": chunk_index,
                    "parent_id": _content_hash(f"{path_str}::{_h2_breadcrumb(meta['section'])}"),
                    "content_hash": _content_hash(document),
                },
            }
        )
    return chunks


def load_doc_chunks(
    docs_dir: Path = DOCS_DIR,
    docs_src_dir: Path = DOCS_SRC_DIR,
    base_url: str = settings.docs_base_url,
    limit: int | None = None,
) -> Iterator[dict]:
    """Load, normalize, and chunk every doc page under docs_dir.

    `limit` caps how many doc pages are loaded (see docs_loader.iter_raw_docs).
    """
    for doc in load_normalized_docs(docs_dir, docs_src_dir, limit=limit):
        yield from doc_to_chunks(doc, base_url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    all_chunks = list(load_doc_chunks())
    token_counts = [_estimate_tokens(c["document"]) for c in all_chunks]
    print(f"Built {len(all_chunks)} chunks from docs")
    print(
        f"tokens/chunk: min={min(token_counts)} "
        f"avg={sum(token_counts) / len(token_counts):.0f} max={max(token_counts)}"
    )
