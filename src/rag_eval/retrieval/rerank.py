"""Cross-encoder reranking stage.

Uses `fastembed.rerank.cross_encoder.TextCrossEncoder` (ONNX, ~90MB, no
torch) rather than `sentence-transformers`: sentence-transformers pulls
torch and roughly triples the deploy image size, breaking the
single-container target (CLAUDE.md, docs/plan.md Phase 6). Do not swap this
for a torch-based reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_eval.retrieval.base import Candidate

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]: ...


@dataclass
class FastEmbedReranker:
    """Lazily loads the ONNX model on first use, not at construction, so
    building a pipeline (e.g. in tests with rerank disabled) never touches
    the model cache or network."""

    model_name: str = DEFAULT_RERANK_MODEL
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if not candidates:
            return []
        model = self._get_model()
        scores = list(model.rerank(query, [c.content for c in candidates]))
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

        out: list[Candidate] = []
        for rank, i in enumerate(order[:top_n], start=1):
            candidate = candidates[i]
            candidate.scores["rerank"] = float(scores[i])
            candidate.ranks["rerank"] = rank
            if "rerank" not in candidate.stages:
                candidate.stages.append("rerank")
            candidate.final_score = float(scores[i])
            out.append(candidate)
        return out
