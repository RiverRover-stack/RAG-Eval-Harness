"""Judge-free retrieval metrics: pure functions, no I/O, no LLM in the
loop. This is the primary number the README leads with, precisely because
nothing here can hallucinate -- a chunk id is either in the gold set or it
isn't.

Every function takes `retrieved` as an ordered list of chunk ids (rank 0
first) and `gold` as a set of chunk ids. Relevance is binary: a retrieved
chunk is either gold or it isn't, there's no partial credit. That's a
simplification (C3 in docs/plan.md notes graded relevance is future work),
but it keeps the metric honest and easy to hand-verify.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field


def _top_k(retrieved: Sequence[str], k: int) -> Sequence[str]:
    return retrieved[:k]


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of gold chunks that appear anywhere in the top k. 0.0 when
    gold is empty -- an item with no resolvable gold can't contribute a
    recall signal either way, so callers should exclude it upstream rather
    than let it silently count as a perfect (or zero) score."""
    if not gold:
        return 0.0
    hit = len(set(_top_k(retrieved, k)) & gold)
    return hit / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    top = _top_k(retrieved, k)
    if not top:
        return 0.0
    hit = len(set(top) & gold)
    return hit / len(top)


def hit_rate_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """1.0 if at least one gold chunk lands in the top k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if set(_top_k(retrieved, k)) & gold else 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """1 / (rank of the first gold hit), 1-indexed. 0.0 if nothing gold is
    retrieved at all."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG. The ideal ranking places min(k, |gold|) gold
    hits in the first min(k, |gold|) ranks, which gives IDCG a closed form
    instead of needing to actually sort anything."""
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(_top_k(retrieved, k), start=1)
        if chunk_id in gold
    )
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


@dataclass
class AggregateMetrics:
    n: int
    k_values: list[int]
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    # 95% bootstrap CIs, keyed the same way the point estimates are:
    # "recall_at_5", "mrr", "ndcg_at_10", ...
    cis: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "k_values": self.k_values,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "cis": self.cis,
        }


def bootstrap_ci(
    values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of `values`. n=1000 resamples,
    seeded so a run's reported CI is reproducible, not a different number
    on every `rag-eval eval run`."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    lo_idx = int((alpha / 2) * n)
    hi_idx = int((1 - alpha / 2) * n) - 1
    lo_idx = max(0, min(lo_idx, n - 1))
    hi_idx = max(0, min(hi_idx, n - 1))
    return (means[lo_idx], means[hi_idx])


def paired_bootstrap(
    base: Sequence[float], cand: Sequence[float], n: int = 1000, seed: int = 0
) -> float:
    """Fraction of bootstrap resamples where cand's mean beats base's mean
    -- a paired significance check for "did the candidate config actually
    win," resampling item indices jointly so paired items stay paired."""
    if len(base) != len(cand):
        raise ValueError("paired_bootstrap requires equal-length, item-aligned sequences")
    if not base:
        return 0.0

    rng = random.Random(seed)
    size = len(base)
    wins = 0
    for _ in range(n):
        idxs = [rng.randrange(size) for _ in range(size)]
        base_mean = sum(base[i] for i in idxs) / size
        cand_mean = sum(cand[i] for i in idxs) / size
        if cand_mean > base_mean:
            wins += 1
    return wins / n


def aggregate(
    per_item: list[tuple[Sequence[str], set[str]]],
    k_values: Sequence[int],
    *,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> AggregateMetrics:
    """per_item: list of (retrieved_chunk_ids, gold_chunk_ids) pairs, one
    per eval item. Items with empty gold are excluded from every metric --
    see recall_at_k's docstring for why."""
    scored_items = [(retrieved, gold) for retrieved, gold in per_item if gold]
    n = len(scored_items)
    result = AggregateMetrics(n=n, k_values=list(k_values))
    if n == 0:
        return result

    mrr_values = [reciprocal_rank(retrieved, gold) for retrieved, gold in scored_items]
    result.mrr = sum(mrr_values) / n
    lo, hi = bootstrap_ci(mrr_values, n=bootstrap_n, seed=seed)
    result.cis["mrr"] = (lo, hi)

    for k in k_values:
        recall_values = [recall_at_k(r, g, k) for r, g in scored_items]
        precision_values = [precision_at_k(r, g, k) for r, g in scored_items]
        hit_values = [hit_rate_at_k(r, g, k) for r, g in scored_items]
        ndcg_values = [ndcg_at_k(r, g, k) for r, g in scored_items]

        result.recall_at_k[k] = sum(recall_values) / n
        result.precision_at_k[k] = sum(precision_values) / n
        result.hit_rate_at_k[k] = sum(hit_values) / n
        result.ndcg_at_k[k] = sum(ndcg_values) / n

        result.cis[f"recall_at_{k}"] = bootstrap_ci(recall_values, n=bootstrap_n, seed=seed)
        result.cis[f"precision_at_{k}"] = bootstrap_ci(precision_values, n=bootstrap_n, seed=seed)
        result.cis[f"hit_rate_at_{k}"] = bootstrap_ci(hit_values, n=bootstrap_n, seed=seed)
        result.cis[f"ndcg_at_{k}"] = bootstrap_ci(ndcg_values, n=bootstrap_n, seed=seed)

    return result
