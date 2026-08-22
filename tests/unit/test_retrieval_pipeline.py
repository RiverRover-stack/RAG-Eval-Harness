"""RetrievalPipeline orchestration, exercised entirely through constructor
injection (CLAUDE.md: prefer constructor injection over patching) -- no
stage here ever touches a real Chroma collection, fastembed model, or LLM.
"""

import pytest

from rag_eval.retrieval.base import Candidate
from rag_eval.retrieval.pipeline import RetrievalPipeline


def _cand(chunk_id: str, source_type: str = "docs", stage: str = "dense", score: float = 1.0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        url=f"https://x/{chunk_id}",
        title="t",
        source_type=source_type,
        scores={stage: score},
        stages=[stage],
        final_score=score,
    )


class SpySearcher:
    def __init__(self, candidates: list[Candidate]):
        self.candidates = candidates
        self.calls: list[tuple] = []

    def search(self, query_text, k, deny_ids=frozenset()):
        self.calls.append((query_text, k, deny_ids))
        return [c for c in self.candidates if c.chunk_id not in deny_ids]


class SpyReranker:
    def __init__(self):
        self.calls: list[tuple] = []

    def rerank(self, query, candidates, top_n):
        self.calls.append((query, len(candidates), top_n))
        return candidates[:top_n]


class SpyRewriter:
    def __init__(self, rewrites: list[str]):
        self.rewrites = rewrites
        self.calls: list[tuple] = []

    def rewrite(self, query, n):
        self.calls.append((query, n))
        return self.rewrites[:n]


class SpyExpander:
    def __init__(self):
        self.calls = 0

    def expand(self, candidates):
        self.calls += 1
        return candidates


def test_dense_only_pipeline_never_touches_unwired_stages():
    dense = SpySearcher([_cand("a", score=0.9), _cand("b", score=0.5)])
    pipeline = RetrievalPipeline(top_k=5, candidates_k=10, dense=dense)

    result = pipeline.retrieve("q")

    assert [c.chunk_id for c in result.candidates] == ["a", "b"]
    assert "bm25" not in result.stage_counts
    assert "rerank" not in result.stage_timings
    assert "query_rewrite" not in result.stage_timings
    assert "expand" not in result.stage_timings


def test_bm25_only_pipeline_has_no_dense_stage():
    bm25 = SpySearcher([_cand("a", source_type="docs", stage="bm25")])
    pipeline = RetrievalPipeline(top_k=5, candidates_k=10, dense=None, bm25=bm25)

    result = pipeline.retrieve("q")

    assert [c.chunk_id for c in result.candidates] == ["a"]
    assert "dense" not in result.stage_counts
    assert "dense" not in result.stage_timings


def test_deny_ids_reach_dense_and_bm25():
    dense = SpySearcher([_cand("a")])
    bm25 = SpySearcher([_cand("b", stage="bm25")])
    pipeline = RetrievalPipeline(top_k=5, candidates_k=10, dense=dense, bm25=bm25)

    pipeline.retrieve("q", deny_ids={"x"})

    assert dense.calls[0][2] == {"x"}
    assert bm25.calls[0][2] == {"x"}


def test_reranker_only_called_when_wired():
    dense = SpySearcher([_cand("a"), _cand("b")])
    reranker = SpyReranker()
    pipeline = RetrievalPipeline(top_k=5, candidates_k=10, dense=dense, reranker=reranker, rerank_top_n=1)

    result = pipeline.retrieve("q")

    assert len(reranker.calls) == 1
    assert len(result.candidates) == 1
    assert result.stage_counts["rerank"] == 1


def test_rerank_not_called_when_not_wired():
    dense = SpySearcher([_cand("a"), _cand("b")])
    pipeline = RetrievalPipeline(top_k=5, candidates_k=10, dense=dense, reranker=None)

    result = pipeline.retrieve("q")

    assert "rerank" not in result.stage_counts
    assert len(result.candidates) == 2


def test_expander_only_called_when_wired():
    dense = SpySearcher([_cand("a")])
    expander = SpyExpander()
    pipeline = RetrievalPipeline(top_k=5, candidates_k=10, dense=dense, expander=None)
    pipeline.retrieve("q")
    assert expander.calls == 0  # never wired in -- can't have been called

    pipeline_with_expand = RetrievalPipeline(top_k=5, candidates_k=10, dense=dense, expander=expander)
    pipeline_with_expand.retrieve("q")
    assert expander.calls == 1


def test_rewriter_feeds_dense_with_original_plus_rewritten_queries():
    dense = SpySearcher([_cand("a")])
    rewriter = SpyRewriter(["hypothetical answer"])
    pipeline = RetrievalPipeline(
        top_k=5, candidates_k=10, dense=dense, rewriter=rewriter, rewrite_n=1
    )

    result = pipeline.retrieve("original question")

    assert rewriter.calls == [("original question", 1)]
    queried_texts = [call[0] for call in dense.calls]
    assert queried_texts == ["original question", "hypothetical answer"]
    assert result.rewritten_queries == ["hypothetical answer"]


def test_per_source_caps_applied_after_fusion():
    dense = SpySearcher(
        [
            _cand("d1", source_type="docs", score=0.9),
            _cand("d2", source_type="docs", score=0.8),
            _cand("d3", source_type="docs", score=0.7),
            _cand("q1", source_type="discussions", score=0.6),
        ]
    )
    pipeline = RetrievalPipeline(
        top_k=5, candidates_k=10, dense=dense, per_source_caps={"docs": 1, "discussions": 1}
    )

    result = pipeline.retrieve("q")

    assert [c.chunk_id for c in result.candidates] == ["d1", "q1"]


def test_from_config_rejects_config_with_no_retrieval_stage_enabled():
    from rag_eval.config.run_config import RunConfig

    cfg = RunConfig(name="t")
    cfg = cfg.model_copy(
        update={
            "retrieval": cfg.retrieval.model_copy(
                update={"dense": cfg.retrieval.dense.model_copy(update={"enabled": False})}
            )
        }
    )
    with pytest.raises(ValueError, match="at least one retrieval stage"):
        RetrievalPipeline.from_config(cfg)
