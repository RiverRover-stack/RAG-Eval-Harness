"""
Turn normalized FastAPI doc pages (see docs_loader.py) into embeddable
chunks, structurally split on `##`/`###` headings.

Metadata shape is kept consistent with the discussions chunker
(chunker.py): source_type, title, url, content_hash are common to every
chunk regardless of source; section/path/chunk_index are docs-only, since
GitHub Discussions answers aren't broken into sub-sections (see README TODO
-- that assumption hasn't been validated against real discussion bodies
yet, only asserted here).

Chunking strategy:
  1. Split each page into sections at `##`/`###` headings (never at `#`,
     the page's own title, or `####`+, which stay folded into their
     nearest `##`/`###` ancestor's section).
  2. Within a section, split into atomic blocks at blank lines, treating
     each fenced ```code``` block as a single atomic block that is never
     split.
  3. Greedily pack those blocks into chunks targeting 400-600 tokens
     (approximated at ~4 chars/token -- there's no exact tokenizer here
     since embeddings come from a local Ollama model, not an OpenAI-style
     tokenizer). The 400 floor is prioritized over the 600 ceiling: a
     chunk keeps absorbing blocks until it has *reached* 400 tokens, even
     if that means overshooting 600 for one chunk. The only hard
     invariant is that a fenced code block is never split to hit either
     bound.

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

logger = logging.getLogger(__name__)

TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 600
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count, ~4 chars/token (no exact tokenizer for the local
    embedding model)."""
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


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
# 2. Split a section into atomic blocks (paragraphs / fenced code)
# ---------------------------------------------------------------------------


def _split_into_blocks(lines: list[str]) -> list[tuple[str, bool]]:
    """Split a section's lines into (text, is_code) blocks: each fenced
    ```code``` block is one atomic block, each blank-line-delimited chunk
    of prose (which may itself be a long, blank-line-free list) is another."""
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_fence = False

    def flush(is_code: bool = False) -> None:
        if buf and any(line.strip() for line in buf):
            blocks.append(("\n".join(buf).strip("\n"), is_code))
        buf.clear()

    for line in lines:
        if line.strip().startswith(_FENCE_PREFIX):
            if not in_fence:
                flush()
                buf.append(line)
                in_fence = True
            else:
                buf.append(line)
                flush(is_code=True)
                in_fence = False
            continue

        if in_fence:
            buf.append(line)
            continue

        if not line.strip():
            flush()
            continue

        buf.append(line)

    flush()
    return blocks


def _split_oversized_block(block: str) -> list[str]:
    """Split a non-code block that alone exceeds the chunk cap (e.g. a long,
    blank-line-free bullet list of PR links in release-notes.md) by line, so
    no single unit blows past the target -- unlike fenced code, prose/list
    text has no atomicity requirement to protect."""
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for line in block.splitlines():
        line_tokens = _estimate_tokens(line)
        if buf and buf_tokens + line_tokens > TARGET_MAX_TOKENS:
            parts.append("\n".join(buf))
            buf = [line]
            buf_tokens = line_tokens
        else:
            buf.append(line)
            buf_tokens += line_tokens

    if buf:
        parts.append("\n".join(buf))
    return parts


def _atomic_blocks(lines: list[str]) -> list[str]:
    """The final list of unsplittable text units for a section: fenced code
    blocks stay whole no matter their size; oversized prose/list blocks are
    broken up by line first."""
    result: list[str] = []
    for text, is_code in _split_into_blocks(lines):
        if not is_code and _estimate_tokens(text) > TARGET_MAX_TOKENS:
            result.extend(_split_oversized_block(text))
        else:
            result.append(text)
    return result


# ---------------------------------------------------------------------------
# 3. Greedily pack blocks into token-bounded chunks
# ---------------------------------------------------------------------------


def _pack_blocks(blocks: list[str]) -> list[str]:
    """Pack atomic blocks into chunks, prioritizing the 400-token floor
    over the 600-token ceiling (see module docstring). A fenced code block
    that alone exceeds the cap still forms its own oversized chunk, since
    it's never split."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = _estimate_tokens(block)
        if current and current_tokens >= TARGET_MIN_TOKENS and (
            current_tokens + block_tokens > TARGET_MAX_TOKENS
        ):
            chunks.append("\n\n".join(current))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


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

    chunks: list[dict] = []
    for section in sections:
        blocks = _atomic_blocks(section.lines)
        if not blocks:
            continue
        for chunk_index, chunk_text in enumerate(_pack_blocks(blocks)):
            chunk_id = _content_hash(f"{path_str}::{section.breadcrumb}::{chunk_index}")
            chunks.append(
                {
                    "id": chunk_id,
                    "document": chunk_text,
                    "metadata": {
                        "source_type": "docs",
                        "title": page_title,
                        "section": section.breadcrumb,
                        "path": path_str,
                        "url": _doc_url(path_str, section.anchor, base_url),
                        "chunk_index": chunk_index,
                        "content_hash": _content_hash(chunk_text),
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
