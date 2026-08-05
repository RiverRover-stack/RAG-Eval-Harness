"""Turn raw DiscussionQA records into embeddable text chunks.

Short answers stay one chunk per answer body, with the question title
carried as metadata for citation. Longer answers are packed into
token-bounded chunks through the shared packer (ingestion/packing.py,
also used by docs_chunker.py) instead of becoming a single oversized
chunk -- a multi-thousand-char answer has a blurry averaged embedding and
floods the context window when retrieved.

Metadata shape here is kept consistent with the docs chunker (docs_chunker.py),
so both sources can share one Chroma collection and be filtered/reported on by
`source_type`. See project notes on the chunk metadata schema: source_type,
title, url, chunk_index, parent_id, content_hash are common to every chunk
regardless of source; section/path are docs-only.
"""

import hashlib

from rag_eval.common.schemas import DiscussionQA
from rag_eval.ingestion.packing import (
    TARGET_MAX_TOKENS,
    _atomic_blocks,
    _estimate_tokens,
    _pack_blocks,
)


def _content_hash(text: str) -> str:
    """Hash of the chunk's embedded text, used to skip re-embedding unchanged
    chunks on re-ingestion (id staying the same only means same slot, not
    same content)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def qa_to_chunks(qa: DiscussionQA) -> list[dict]:
    """Return Chroma-ready {id, document, metadata} dicts for one Q&A pair.

    A short answer stays a single chunk keyed by the GitHub GraphQL
    discussion node ID -- already globally unique and stable, so it
    doubles as the idempotent dedup key with no extra hashing needed. A
    longer answer is packed into several chunks first, each keyed by
    `<discussion_id>::<chunk_index>` instead.
    """
    answer = qa.answer_body
    if _estimate_tokens(answer) <= TARGET_MAX_TOKENS:
        texts = [answer]
    else:
        texts = _pack_blocks(_atomic_blocks(answer.splitlines()))

    parent_id = _content_hash(qa.discussion_id)
    single = len(texts) == 1
    chunks = []
    for chunk_index, text in enumerate(texts):
        chunk_id = qa.discussion_id if single else f"{qa.discussion_id}::{chunk_index}"
        chunks.append(
            {
                "id": chunk_id,
                "document": text,
                "metadata": {
                    "source_type": "discussion",
                    "title": qa.title,
                    "question": qa.question_body[:500],
                    "url": qa.url,
                    "category": qa.category or "",
                    "chunk_index": chunk_index,
                    "parent_id": parent_id,
                    "content_hash": _content_hash(text),
                },
            }
        )
    return chunks
