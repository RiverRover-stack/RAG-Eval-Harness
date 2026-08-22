"""Dense-vector search stage.

Embeds the query once and queries each configured source's Chroma
collection via `vector_store.query`, converting hits into `Candidate`s
tagged with a "dense" score/rank. `query_fn` is injected (defaults to
`vector_store.query`) so tests can exercise ranking/dedup logic without a
real Chroma collection or embedder (CLAUDE.md: prefer constructor
injection over patching).
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_eval.retrieval.base import Candidate

if TYPE_CHECKING:
    from rag_eval.providers.base import EmbeddingProvider

QueryFn = Callable[..., list[dict]]


@dataclass
class DenseSearcher:
    sources: list[str]
    embedder: EmbeddingProvider
    query_fn: QueryFn

    def search(
        self, query_text: str, k: int, deny_ids: AbstractSet[str] = frozenset()
    ) -> list[Candidate]:
        query_vec = self.embedder.embed_query(query_text)
        wanted = k + len(deny_ids)
        # Tag each hit with the source it came from -- this becomes
        # Candidate.source_type, which must match cfg.retrieval.per_source_caps'
        # keys ("docs"/"discussions") and cfg.corpus.sources, NOT the chunk
        # metadata's own "source_type" field (ingestion writes "discussion",
        # singular, for the discussions collection; using that here would
        # silently break per-source caps for that source).
        hits: list[tuple[str, dict]] = []
        for source in self.sources:
            hits.extend((source, hit) for hit in self.query_fn(query_vec, source, self.embedder, k=wanted))
        hits.sort(key=lambda item: item[1]["score"], reverse=True)

        candidates: list[Candidate] = []
        rank = 0
        for source, hit in hits:
            if hit["id"] in deny_ids:
                continue
            rank += 1
            if rank > k:
                break
            meta = hit["metadata"] or {}
            candidates.append(
                Candidate(
                    chunk_id=hit["id"],
                    content=hit["content"],
                    url=meta.get("url", ""),
                    title=meta.get("title", ""),
                    source_type=source,
                    scores={"dense": hit["score"]},
                    ranks={"dense": rank},
                    stages=["dense"],
                    final_score=hit["score"],
                )
            )
        return candidates
