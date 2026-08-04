"""Turn raw DiscussionQA records into embeddable text chunks.

Keep this simple to start: one chunk per answer body, with the question
title carried as metadata for citation. Swap in a token-aware splitter
later if answers turn out to be long.

Metadata shape here is kept consistent with the docs chunker (docs_chunker.py,
once it exists), so both sources can share one Chroma collection and be
filtered/reported on by `source_type`. See project notes on the chunk
metadata schema: source_type, title, url, content_hash are common to every
chunk regardless of source; section/path/chunk_index are docs-only.
"""

import hashlib

from rag_eval.common.schemas import DiscussionQA


def _content_hash(text: str) -> str:
    """Hash of the chunk's embedded text, used to skip re-embedding unchanged
    chunks on re-ingestion (id staying the same only means same slot, not
    same content)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def qa_to_chunks(qa: DiscussionQA) -> list[dict]:
    """Return Chroma-ready {id, document, metadata} dicts for one Q&A pair.

    id is the GitHub GraphQL discussion node ID, which is already globally
    unique and stable, so it doubles as the idempotent dedup key with no
    extra hashing needed.
    """
    return [
        {
            "id": qa.discussion_id,
            "document": qa.answer_body,
            "metadata": {
                "source_type": "discussion",
                "title": qa.title,
                "question": qa.question_body[:500],
                "url": qa.url,
                "category": qa.category or "",
                "content_hash": _content_hash(qa.answer_body),
            },
        }
    ]
