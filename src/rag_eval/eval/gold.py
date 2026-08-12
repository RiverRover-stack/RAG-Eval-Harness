"""Gold labels, keyed by URL rather than chunk id (CLAUDE.md: chunk ids are
content hashes and shift the moment the chunker changes, but a doc section
or discussion answer keeps its URL across re-chunks).

`resolve_gold_chunks` is what turns a URL into the chunk id(s) the current
index actually holds for it, re-run at eval-load time rather than baked
into the JSONL, so a chunker change surfaces as "gold URL doesn't resolve"
instead of a silently stale chunk id.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalItem(BaseModel):
    id: str
    dataset: str
    question: str
    ground_truth: str | None = None
    ground_truth_raw: str | None = None
    gold_urls: list[str] = []
    gold_chunk_ids: list[str] = []  # cached hint, re-resolved at load, not trusted as-is
    gold_granularity: Literal["anchor", "page"] = "anchor"
    exclude_chunk_ids: list[str] = []  # self-retrieval leakage guard
    provenance: str = ""
    # filled in by eval/review.py's human checkpoints; None means "not
    # reviewed yet", not "rejected" -- most items in docs_synth_v1 stay None
    # forever, since only a sample gets human-reviewed.
    verified: Literal["yes", "no", "edited"] | None = None
    verified_at: str | None = None


@dataclass
class GoldIndex:
    # exact chunk url -> the chunk id(s) that carry it (almost always one,
    # but nothing stops two chunks sharing a url after the undersized-merge
    # pass, so this stays a set)
    by_url: dict[str, set[str]]
    # page (url with any #fragment stripped) -> every chunk id on that page,
    # used only when an exact anchor match fails to resolve
    by_page: dict[str, set[str]]


def _page_of(url: str) -> str:
    return url.split("#", 1)[0]


def build_gold_index(chunks: Iterable[dict]) -> GoldIndex:
    """chunks: {id, document, metadata} dicts as produced by
    docs_chunker.load_doc_chunks / chunker.qa_to_chunks -- rebuilt straight
    from the corpus snapshot rather than read back out of Chroma, so gold
    resolution works without a live index."""
    by_url: dict[str, set[str]] = {}
    by_page: dict[str, set[str]] = {}
    for chunk in chunks:
        url = chunk["metadata"].get("url", "")
        if not url:
            continue
        chunk_id = chunk["id"]
        by_url.setdefault(url, set()).add(chunk_id)
        by_page.setdefault(_page_of(url), set()).add(chunk_id)
    return GoldIndex(by_url=by_url, by_page=by_page)


def resolve_gold_chunks(item: EvalItem, index: GoldIndex) -> EvalItem:
    """Resolve `item.gold_urls` against the current corpus, returning a copy
    of `item` with `gold_chunk_ids` and `gold_granularity` filled in.

    A URL that matches a chunk's exact url resolves at anchor granularity.
    One that only matches at the page level (either because it was recorded
    without a fragment, or because its fragment doesn't exist as any
    chunk's url) falls back to every chunk on that page and is flagged
    `gold_granularity: page` -- reported separately rather than folded into
    the headline number, since page-level gold inflates recall.
    """
    chunk_ids: set[str] = set()
    used_page_fallback = False
    for url in item.gold_urls:
        if url in index.by_url:
            chunk_ids |= index.by_url[url]
            continue
        page = _page_of(url)
        if page in index.by_page:
            chunk_ids |= index.by_page[page]
            used_page_fallback = True
        else:
            logger.warning(
                "gold URL does not resolve to any chunk (item=%s dataset=%s): %s",
                item.id,
                item.dataset,
                url,
            )

    granularity: Literal["anchor", "page"] = "page" if used_page_fallback else "anchor"
    return item.model_copy(
        update={"gold_chunk_ids": sorted(chunk_ids), "gold_granularity": granularity}
    )


def resolve_all(items: Iterable[EvalItem], index: GoldIndex) -> list[EvalItem]:
    return [resolve_gold_chunks(item, index) for item in items]
