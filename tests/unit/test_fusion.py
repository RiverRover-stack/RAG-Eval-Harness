from rag_eval.retrieval.base import Candidate
from rag_eval.retrieval.fusion import apply_per_source_caps, reciprocal_rank_fusion


def _cand(chunk_id: str, source_type: str = "docs", stage: str = "dense", score: float = 1.0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        url=f"https://x/{chunk_id}",
        title="t",
        source_type=source_type,
        scores={stage: score},
        ranks={},
        stages=[stage],
        final_score=score,
    )


def test_rrf_dedups_a_chunk_appearing_in_multiple_rankings():
    dense_ranking = [_cand("a", stage="dense"), _cand("b", stage="dense")]
    bm25_ranking = [_cand("a", stage="bm25"), _cand("c", stage="bm25")]

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])

    assert sorted(c.chunk_id for c in fused) == ["a", "b", "c"]
    a = next(c for c in fused if c.chunk_id == "a")
    assert set(a.stages) == {"dense", "bm25", "fusion"}
    assert set(a.scores) == {"dense", "bm25", "rrf"}


def test_rrf_score_is_sum_of_1_over_k_plus_position():
    dense_ranking = [_cand("a"), _cand("b")]  # a: position 1, b: position 2
    bm25_ranking = [_cand("b"), _cand("a")]  # b: position 1, a: position 2

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60)

    a = next(c for c in fused if c.chunk_id == "a")
    b = next(c for c in fused if c.chunk_id == "b")
    assert a.scores["rrf"] == 1 / 61 + 1 / 62
    assert b.scores["rrf"] == 1 / 62 + 1 / 61
    assert a.scores["rrf"] == b.scores["rrf"]  # symmetric positions -> tied


def test_rrf_orders_by_combined_score_descending():
    # "a" appears near the top of both rankings, "b" only appears once and late
    dense_ranking = [_cand("a"), _cand("z1"), _cand("z2")]
    bm25_ranking = [_cand("a"), _cand("z3"), _cand("b")]

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])

    assert fused[0].chunk_id == "a"
    assert fused[0].ranks["fusion"] == 1
    assert fused[-1].chunk_id == "b"


def test_rrf_single_ranking_preserves_relative_order():
    ranking = [_cand("a"), _cand("b"), _cand("c")]
    fused = reciprocal_rank_fusion([ranking])
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_apply_per_source_caps_limits_each_source():
    candidates = [
        _cand("d1", source_type="docs"),
        _cand("d2", source_type="docs"),
        _cand("d3", source_type="docs"),
        _cand("q1", source_type="discussions"),
        _cand("q2", source_type="discussions"),
    ]

    capped = apply_per_source_caps(candidates, {"docs": 2, "discussions": 1})

    assert [c.chunk_id for c in capped] == ["d1", "d2", "q1"]


def test_apply_per_source_caps_leaves_unlisted_sources_uncapped():
    candidates = [_cand("a", source_type="other"), _cand("b", source_type="other")]
    capped = apply_per_source_caps(candidates, {"docs": 1})
    assert [c.chunk_id for c in capped] == ["a", "b"]
