"""Reciprocal rank fusion and per-source caps.

Why RRF and not score normalization: dense cosine scores and BM25 scores
live in unrelated, differently-shaped distributions (and, before this
phase, so did dense scores across the docs vs discussions collections --
long discussion answers systematically beat terse docs chunks under raw
score sorting). Rank-based fusion sidesteps that entirely: it only cares
about a candidate's *position* in each stage's ranking, never the raw score
value, so it is distribution-free by construction (docs/plan.md Phase 6).
"""

from __future__ import annotations

from rag_eval.retrieval.base import Candidate


def reciprocal_rank_fusion(rankings: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    """Merge one or more stage rankings into one, deduped by chunk_id.
    `k` is the RRF rank-damping constant; a candidate's contribution from
    each ranking is `1 / (k + position)`, position being 1-based."""
    merged: dict[str, Candidate] = {}
    rrf_scores: dict[str, float] = {}

    for ranking in rankings:
        for position, candidate in enumerate(ranking, start=1):
            rrf_scores[candidate.chunk_id] = rrf_scores.get(candidate.chunk_id, 0.0) + 1.0 / (
                k + position
            )
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = Candidate(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    url=candidate.url,
                    title=candidate.title,
                    source_type=candidate.source_type,
                    scores=dict(candidate.scores),
                    ranks=dict(candidate.ranks),
                    stages=list(candidate.stages),
                )
            else:
                existing.scores.update(candidate.scores)
                existing.ranks.update(candidate.ranks)
                for stage in candidate.stages:
                    if stage not in existing.stages:
                        existing.stages.append(stage)

    for chunk_id, candidate in merged.items():
        candidate.scores["rrf"] = rrf_scores[chunk_id]
        candidate.final_score = rrf_scores[chunk_id]
        if "fusion" not in candidate.stages:
            candidate.stages.append("fusion")

    fused = sorted(merged.values(), key=lambda c: c.final_score, reverse=True)
    for rank, candidate in enumerate(fused, start=1):
        candidate.ranks["fusion"] = rank
    return fused


def apply_per_source_caps(candidates: list[Candidate], caps: dict[str, int]) -> list[Candidate]:
    """Keep list order, drop candidates once their source_type has hit its
    cap. A source_type absent from `caps` is uncapped."""
    counts: dict[str, int] = {}
    out: list[Candidate] = []
    for candidate in candidates:
        cap = caps.get(candidate.source_type)
        if cap is not None:
            if counts.get(candidate.source_type, 0) >= cap:
                continue
            counts[candidate.source_type] = counts.get(candidate.source_type, 0) + 1
        out.append(candidate)
    return out
