"""Parent/section expansion stage.

Needs no new storage: `parent_id` and `chunk_index` are already written
into every chunk's metadata by the chunkers (docs/plan.md Phase 6), so a
candidate's siblings are just a Chroma `where={"parent_id": ...}` query
against the same collection it came from. Siblings are joined in
`chunk_index` order up to `max_tokens`, reusing the ingestion side's own
token estimate so the budget means the same thing it did when chunks were
originally packed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_eval.ingestion.packing import _estimate_tokens
from rag_eval.retrieval.base import Candidate

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


def _chunk_index(meta: object) -> int:
    value = meta.get("chunk_index", 0) if isinstance(meta, dict) else 0
    return int(value) if isinstance(value, (int, float)) else 0


@dataclass
class ParentExpander:
    collections: dict[str, Collection] = field(default_factory=dict)
    max_tokens: int = 1200

    def expand(self, candidates: list[Candidate]) -> list[Candidate]:
        for candidate in candidates:
            collection = self.collections.get(candidate.source_type)
            if collection is not None:
                self._expand_one(candidate, collection)
        return candidates

    def _expand_one(self, candidate: Candidate, collection: Collection) -> None:
        anchor = collection.get(ids=[candidate.chunk_id], include=["metadatas"])
        metas = anchor["metadatas"] or []
        if not metas:
            return
        parent_id = (metas[0] or {}).get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            return

        siblings = collection.get(where={"parent_id": parent_id}, include=["documents", "metadatas"])
        rows = list(
            zip(siblings["ids"] or [], siblings["documents"] or [], siblings["metadatas"] or [])
        )
        rows.sort(key=lambda row: _chunk_index(row[2]))

        merged: list[str] = []
        total_tokens = 0
        for _chunk_id, content, _meta in rows:
            tokens = _estimate_tokens(content)
            if merged and total_tokens + tokens > self.max_tokens:
                break
            merged.append(content)
            total_tokens += tokens

        if len(merged) > 1:
            candidate.content = "\n\n".join(merged)
            if "expand" not in candidate.stages:
                candidate.stages.append("expand")
