"""If this file is wrong, every metric this project reports is wrong --
so several cases here are worked out by hand from the definition, not by
calling the function under test with different inputs."""

import math

import pytest

from rag_eval.eval.retrieval_metrics import (
    aggregate,
    bootstrap_ci,
    hit_rate_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]
GOLD = {"c", "e"}


def test_recall_at_k_basic():
    assert recall_at_k(RETRIEVED, GOLD, k=5) == 1.0
    assert recall_at_k(RETRIEVED, GOLD, k=2) == 0.0
    assert recall_at_k(RETRIEVED, GOLD, k=3) == pytest.approx(0.5)


def test_recall_at_k_when_gold_exceeds_k():
    # 3 gold chunks, only 2 fit in the requested top-k -- recall must stay
    # capped by the full gold set size, not by k.
    retrieved = ["a", "b", "c"]
    gold = {"a", "b", "x"}
    assert recall_at_k(retrieved, gold, k=2) == pytest.approx(2 / 3)


def test_recall_at_k_empty_gold_is_zero():
    assert recall_at_k(RETRIEVED, set(), k=5) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, GOLD, k=5) == pytest.approx(0.4)
    assert precision_at_k(RETRIEVED, GOLD, k=2) == 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k(RETRIEVED, GOLD, k=3) == 1.0
    assert hit_rate_at_k(RETRIEVED, GOLD, k=2) == 0.0


def test_reciprocal_rank_first_hit_at_rank_3():
    assert reciprocal_rank(RETRIEVED, GOLD) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_nothing_gold_retrieved():
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_zero_when_gold_empty():
    assert reciprocal_rank(RETRIEVED, set()) == 0.0


def test_ndcg_at_1_no_hit_in_first_slot():
    # top-1 = ["a"], not gold -> DCG=0 regardless of IDCG.
    assert ndcg_at_k(RETRIEVED, GOLD, k=1) == 0.0


def test_ndcg_at_3_hand_computed():
    # top-3 = [a, b, c]; only "c" (rank 3) is gold.
    # DCG   = 1/log2(3+1) = 1/log2(4) = 0.5
    # IDCG  = best case puts both gold hits in ranks 1,2 (min(k=3, |gold|=2)=2):
    #         1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_k(RETRIEVED, GOLD, k=3) == pytest.approx(expected)
    assert expected == pytest.approx(0.30657, abs=1e-5)


def test_ndcg_perfect_ranking_is_one():
    retrieved = ["c", "e", "a", "b", "d"]
    assert ndcg_at_k(retrieved, GOLD, k=5) == pytest.approx(1.0)


def test_ndcg_empty_gold_is_zero():
    assert ndcg_at_k(RETRIEVED, set(), k=5) == 0.0


def test_bootstrap_ci_is_deterministic_given_a_seed():
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]
    first = bootstrap_ci(values, n=500, seed=42)
    second = bootstrap_ci(values, n=500, seed=42)
    assert first == second


def test_bootstrap_ci_brackets_the_mean_for_constant_values():
    lo, hi = bootstrap_ci([0.7, 0.7, 0.7], n=200, seed=0)
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.7)


def test_bootstrap_ci_single_value_is_a_point():
    assert bootstrap_ci([0.42]) == (0.42, 0.42)


def test_bootstrap_ci_empty_is_zero_zero():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_paired_bootstrap_candidate_always_wins():
    base = [0.1, 0.2, 0.1, 0.3]
    cand = [0.9, 0.95, 0.99, 0.85]
    assert paired_bootstrap(base, cand, n=200, seed=0) == pytest.approx(1.0)


def test_paired_bootstrap_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap([0.1], [0.1, 0.2])


def test_aggregate_excludes_items_with_empty_gold():
    per_item = [
        (["a", "b"], {"a"}),
        (["x", "y"], set()),  # unresolvable gold -- must not count
    ]
    result = aggregate(per_item, k_values=[1, 2])
    assert result.n == 1


def test_aggregate_matches_manual_mrr():
    per_item = [
        (["a", "b", "c"], {"a"}),  # RR = 1
        (["a", "b", "c"], {"b"}),  # RR = 1/2
        (["a", "b", "c"], {"z"}),  # RR = 0
    ]
    result = aggregate(per_item, k_values=[3])
    assert result.n == 3
    assert result.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert "mrr" in result.cis


def test_aggregate_empty_per_item():
    result = aggregate([], k_values=[1, 5])
    assert result.n == 0
    assert result.recall_at_k == {}
