"""Shared shapes for the retrieval stack (docs/plan.md Phase 6).

`RetrievalResult` is simultaneously the frontend trace payload, the
`retrieval.jsonl` row, and the metric input -- one shape, three consumers.
Each `Candidate` accumulates its per-stage scores/ranks as it passes through
the pipeline (dense -> bm25 -> fusion -> rerank -> expand) rather than
being replaced at each step, so the trace panel can show "reranked ^12"
instead of just a final score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    chunk_id: str
    content: str
    url: str
    title: str
    source_type: str
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    stages: list[str] = field(default_factory=list)
    final_score: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    rewritten_queries: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
