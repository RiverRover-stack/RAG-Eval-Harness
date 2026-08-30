"""No-extra-LLM-call groundedness check (docs/plan.md Phase 7,
`GroundednessConfig.mode == "embedding_support"`): does each sentence of the
answer have semantic support in what was actually retrieved?

Pure cosine similarity over the same embedder already used for retrieval --
no torch, no second model, consistent with this project's ONNX-only,
single-container constraint (CLAUDE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_eval.rag.citations import split_sentences

if TYPE_CHECKING:
    from rag_eval.providers.base import EmbeddingProvider
    from rag_eval.retrieval.base import Candidate


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def sentence_support(
    answer: str, candidates: list[Candidate], embedder: EmbeddingProvider
) -> list[SentenceSupport]:
    sentences = [s for s, _, _ in split_sentences(answer)]
    if not sentences or not candidates:
        return [SentenceSupport(sentence=s, score=0.0) for s in sentences]

    sentence_vecs = embedder.embed_documents(sentences)
    candidate_vecs = embedder.embed_documents([c.content for c in candidates])
    return [
        SentenceSupport(sentence=sentence, score=max(_cosine(vec, cvec) for cvec in candidate_vecs))
        for sentence, vec in zip(sentences, sentence_vecs)
    ]


def groundedness_score(supports: list[SentenceSupport]) -> float:
    if not supports:
        return 0.0
    return sum(s.score for s in supports) / len(supports)


def should_abstain(score: float, threshold: float) -> bool:
    return score < threshold
