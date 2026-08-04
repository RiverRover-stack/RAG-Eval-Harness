"""
Load and normalize the FastAPI documentation pages (data/raw/docs +
data/raw/docs_src) into clean markdown text suitable for chunking/embedding.

The FastAPI docs are written for mkdocs-material plus a few custom
extensions that don't render as plain markdown, so each is normalized away
by its own function:

  - `{* path hl[...] *}` snippet macros that pull in a source file from
    docs_src -> resolved into a real fenced code block (resolve_code_macros)
  - `/// tip ... ///` admonitions -> plain prose, e.g. `*Tip* ...`
    (convert_admonitions)
  - `<div class="termy">...</div>` fake-terminal blocks with inline
    `<font>`/`<span>`/`<u>` coloring tags -> plain terminal text
    (strip_termy_tags)
  - `## Title { #anchor-id }` headings -> `## Title`, with the anchor kept
    alongside instead of discarded (clean_headings)

Usage:
    uv run python -m rag_eval.ingestion.docs_loader
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DOCS_DIR = Path("data/raw/docs")
DOCS_SRC_DIR = Path("data/raw/docs_src")


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------
EXCLUDED_DOCS = {"release-notes.md"}

def iter_raw_docs(
    docs_dir: Path = DOCS_DIR, limit: int | None = None
) -> Iterator[tuple[Path, str]]:
    """Yield (path, raw_text) for every .md file under docs_dir.

    `path` is kept relative to docs_dir so it survives as a stable reference
    back to the source page, e.g. for citing retrieved chunks later.

    `limit` caps how many pages are yielded (in sorted path order), so a
    smaller slice of the docs can be indexed for a quick/local eval run.
    """
    count = 0
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name in EXCLUDED_DOCS:
            continue
        if limit and count >= limit:
            return
        raw_text = path.read_text(encoding="utf-8")
        count += 1
        yield path.relative_to(docs_dir), raw_text


# ---------------------------------------------------------------------------
# 1. Code-macro resolution
# ---------------------------------------------------------------------------

_MACRO_RE = re.compile(r"\{\*\s*(?P<path>\S+)(?P<attrs>[^*]*)\*\}")
_LN_RE = re.compile(r"ln\[([\d,:]+)\]")


def _parse_line_ranges(attrs: str) -> list[tuple[int, int]] | None:
    """Parse a `ln[1:9,29:35]` macro attribute into inclusive, 1-indexed
    (start, end) ranges. Returns None if the macro has no `ln[...]` (meaning:
    show the whole file)."""
    m = _LN_RE.search(attrs)
    if not m:
        return None
    ranges = []
    for part in m.group(1).split(","):
        if ":" in part:
            start, end = part.split(":")
            ranges.append((int(start), int(end)))
        else:
            n = int(part)
            ranges.append((n, n))
    return ranges


def _resolve_snippet_path(raw_path: str, raw_root: Path) -> Path:
    """Resolve a macro's `../../docs_src/...`-style path against raw_root
    (the parent of both docs/ and docs_src/).

    The FastAPI docs always write these paths as if the .md file lived two
    levels below the repo root, regardless of how deeply it's actually
    nested (e.g. tutorial/security/first-steps.md still uses `../../`, same
    as tutorial/query-params.md), so the leading `../` segments carry no
    real path information and are simply dropped.
    """
    parts = [p for p in Path(raw_path).parts if p not in ("..", ".")]
    return raw_root.joinpath(*parts)


def resolve_code_macros(text: str, docs_src_root: Path = DOCS_SRC_DIR) -> str:
    """Replace `{* path hl[...] *}` code-snippet macros with real fenced
    code blocks, read from the referenced file in docs_src."""
    raw_root = docs_src_root.parent

    def repl(match: re.Match[str]) -> str:
        rel_path = match.group("path")
        attrs = match.group("attrs")
        snippet_path = _resolve_snippet_path(rel_path, raw_root)

        if not snippet_path.is_file():
            logger.warning("docs_loader: snippet file not found: %s", snippet_path)
            return f"```python\n# snippet not found: {rel_path}\n```"

        lines = snippet_path.read_text(encoding="utf-8").splitlines()
        ranges = _parse_line_ranges(attrs)
        if ranges:
            selected: list[str] = []
            for start, end in ranges:
                selected.extend(lines[start - 1 : end])
            code = "\n".join(selected)
        else:
            code = "\n".join(lines)

        lang = "python" if snippet_path.suffix == ".py" else snippet_path.suffix.lstrip(".")
        return f"```{lang}\n{code}\n```"

    return _MACRO_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# 2. Admonitions
# ---------------------------------------------------------------------------

_ADMONITION_OPEN_RE = re.compile(r"^/{3,}\s*([A-Za-z][\w-]*)\s*(?:\|\s*(.*?)\s*)?$")
_ADMONITION_CLOSE_RE = re.compile(r"^/{3,}\s*$")


def convert_admonitions(text: str) -> str:
    """Turn `/// tip ... ///` admonitions (and nested `//// tab ... ////`
    blocks) into plain prose: a `*Tip*` (or `*Tip: Title*`) marker line
    followed by the block's own content, dropping the fence lines."""
    out_lines: list[str] = []
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()

        if depth > 0 and _ADMONITION_CLOSE_RE.match(stripped):
            depth -= 1
            continue

        open_match = _ADMONITION_OPEN_RE.match(stripped)
        if open_match:
            depth += 1
            kind, title = open_match.groups()
            label = kind.capitalize()
            if title:
                label = f"{label}: {title}"
            out_lines.append(f"*{label}*")
            continue

        out_lines.append(line)

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 3. Termy tags
# ---------------------------------------------------------------------------

_TERMY_BLOCK_RE = re.compile(
    r"<div\s+class=[\"']termy[\"']\s*>\s*\n(?P<body>.*?)\n</div>",
    re.DOTALL,
)
_INLINE_HTML_RE = re.compile(r"</?(?:font|span|u|b|i|div)\b[^>]*>")


def strip_termy_tags(text: str) -> str:
    """Strip `<div class="termy">...</div>` wrappers and the inline HTML
    tags (`<font>`, `<span>`, `<u>`, ...) used to fake terminal colors,
    keeping just the plain terminal text and its output."""

    def repl(match: re.Match[str]) -> str:
        body = _INLINE_HTML_RE.sub("", match.group("body"))
        return html.unescape(body).strip("\n")

    return _TERMY_BLOCK_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# 4. Heading anchors
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*\{\s*#(?P<anchor>[\w-]+)\s*\}\s*$"
)


@dataclass
class HeadingAnchor:
    level: int
    title: str
    anchor: str


def clean_headings(text: str) -> tuple[str, list[HeadingAnchor]]:
    """Strip the `{ #anchor-id }` suffix off headings for display text,
    returning the anchors separately instead of losing them."""
    anchors: list[HeadingAnchor] = []
    out_lines: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            hashes, title, anchor = m.group("hashes"), m.group("title"), m.group("anchor")
            anchors.append(HeadingAnchor(level=len(hashes), title=title, anchor=anchor))
            out_lines.append(f"{hashes} {title}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines), anchors


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class NormalizedDoc:
    path: Path
    raw_text: str
    text: str
    anchors: list[HeadingAnchor] = field(default_factory=list)


def normalize_doc(path: Path, raw_text: str, docs_src_root: Path = DOCS_SRC_DIR) -> NormalizedDoc:
    """Run all four normalization steps over one doc's raw markdown."""
    text = resolve_code_macros(raw_text, docs_src_root)
    text = strip_termy_tags(text)
    text = convert_admonitions(text)
    text, anchors = clean_headings(text)
    return NormalizedDoc(path=path, raw_text=raw_text, text=text, anchors=anchors)


def load_normalized_docs(
    docs_dir: Path = DOCS_DIR,
    docs_src_dir: Path = DOCS_SRC_DIR,
    limit: int | None = None,
) -> Iterator[NormalizedDoc]:
    """Load every .md file under docs_dir and yield it fully normalized."""
    for path, raw_text in iter_raw_docs(docs_dir, limit=limit):
        yield normalize_doc(path, raw_text, docs_src_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = list(load_normalized_docs())
    total_anchors = sum(len(d.anchors) for d in docs)
    print(f"Loaded and normalized {len(docs)} docs ({total_anchors} headings with anchors)")
