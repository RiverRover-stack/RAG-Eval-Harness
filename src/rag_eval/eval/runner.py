"""Orchestrates one eval run: load dataset(s), resolve gold, retrieve,
score, write the run directory. Retrieval-only for now -- generation
scoring and anything past dense-only retrieval (bm25, rerank, query
rewrite, parent expansion) come with the pipeline that implements them
(docs/plan.md Phase 6/7). Asking for those here fails loudly rather than
silently falling back to dense.

Both the corpus-derived gold index and the actual retrieval call are
injectable, so tests can exercise the whole orchestration with fakes and
never touch fastembed or a live Chroma collection (CLAUDE.md: prefer
constructor injection over patching).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict

from rag_eval.config.run_config import RunConfig
from rag_eval.eval.datasets import DEFAULT_EVAL_SETS_DIR, dataset_sha256, load_dataset
from rag_eval.eval.gold import EvalItem, GoldIndex, resolve_gold_chunks
from rag_eval.eval.retrieval_metrics import (
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.runs.manifest import DEFAULT_RUNS_ROOT, RunManifest, new_run, write_manifest


class Candidate(TypedDict):
    chunk_id: str
    url: str
    score: float


RetrieveFn = Callable[[str, RunConfig, "set[str]"], list[Candidate]]

_UNIMPLEMENTED_STAGES = (
    ("bm25", lambda r: r.bm25.enabled),
    ("rerank", lambda r: r.rerank.enabled),
    ("query_rewrite", lambda r: r.query_rewrite.enabled),
    ("parent_expansion", lambda r: r.parent_expansion.enabled),
)


def _check_supported(cfg: RunConfig) -> None:
    if cfg.generation.enabled:
        raise NotImplementedError(
            "generation scoring isn't wired into the runner yet -- set generation.enabled: false"
        )
    if cfg.eval.self_retrieval == "separate_index":
        raise NotImplementedError("eval.self_retrieval: separate_index isn't built yet")
    if not cfg.retrieval.dense.enabled:
        raise NotImplementedError(
            "retrieval.dense.enabled: false -- dense is the only retrieval stage "
            "this runner implements, so disabling it leaves nothing to retrieve "
            "with (docs/plan.md Phase 6 adds bm25/rerank as real alternatives)"
        )
    for stage_name, is_enabled in _UNIMPLEMENTED_STAGES:
        if is_enabled(cfg.retrieval):
            raise NotImplementedError(
                f"retrieval.{stage_name}.enabled: only dense retrieval exists so far "
                "(docs/plan.md Phase 6 adds the rest)"
            )


def build_corpus_gold_index() -> GoldIndex:
    """Rebuild the gold index straight from the committed corpus snapshot --
    no Chroma required, and it can never go stale relative to the index
    since it *is* the source the index gets built from."""
    from rag_eval.eval.gold import build_gold_index
    from rag_eval.ingestion.chunker import qa_to_chunks
    from rag_eval.ingestion.discussions_snapshot import load_snapshot
    from rag_eval.ingestion.docs_chunker import load_doc_chunks

    doc_chunks = list(load_doc_chunks())
    discussion_chunks = [c for qa in load_snapshot() for c in qa_to_chunks(qa)]
    return build_gold_index(doc_chunks + discussion_chunks)


def _combined_corpus_sha() -> str:
    import hashlib

    from rag_eval.ingestion.corpus_sha import discussions_corpus_sha, docs_corpus_sha

    docs_sha = docs_corpus_sha()
    disc_sha = discussions_corpus_sha()
    return hashlib.sha256(f"{docs_sha}:{disc_sha}".encode()).hexdigest()[:12]


def _default_retrieve_fn(cfg: RunConfig) -> RetrieveFn:
    """Dense-only retrieval across every source in cfg.corpus.sources,
    merged by score (docs/plan.md Phase 6 replaces this with RRF fusion +
    rerank). deny_ids is handled by over-fetching and filtering in Python --
    Chroma's query() has no "exclude this id" option."""
    from rag_eval.providers import get_embedder
    from rag_eval.rag.vector_store import get_collection
    from rag_eval.rag.vector_store import query as vector_query

    embedder = get_embedder(cfg.embedding.provider, cfg.embedding.model)

    def retrieve(question: str, run_cfg: RunConfig, deny_ids: set[str]) -> list[Candidate]:
        query_vec = embedder.embed_query(question)
        wanted = run_cfg.retrieval.candidates_k + len(deny_ids)
        hits: list[dict] = []
        for source in run_cfg.corpus.sources:
            collection = get_collection(source, embedder, create=False)
            n = min(wanted, collection.count())
            if n <= 0:
                continue
            hits.extend(vector_query(query_vec, source, embedder, k=n))
        hits.sort(key=lambda h: h["score"], reverse=True)
        candidates = [
            Candidate(chunk_id=h["id"], url=h["metadata"].get("url", ""), score=h["score"])
            for h in hits
            if h["id"] not in deny_ids
        ]
        return candidates[: run_cfg.retrieval.candidates_k]

    return retrieve


def _score_item(
    item: EvalItem, candidates: list[Candidate], k_values: Sequence[int]
) -> dict[str, float]:
    retrieved_ids = [c["chunk_id"] for c in candidates]
    gold = set(item.gold_chunk_ids)
    metrics: dict[str, float] = {"mrr": reciprocal_rank(retrieved_ids, gold)}
    for k in k_values:
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gold, k)
        metrics[f"precision_at_{k}"] = precision_at_k(retrieved_ids, gold, k)
        metrics[f"hit_rate_at_{k}"] = hit_rate_at_k(retrieved_ids, gold, k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_ids, gold, k)
    return metrics


def run_experiment(
    cfg: RunConfig,
    config_path: str | Path,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    eval_sets_dir: Path = DEFAULT_EVAL_SETS_DIR,
    gold_index: GoldIndex | None = None,
    retrieve_fn: RetrieveFn | None = None,
    corpus_sha: str | None = None,
) -> RunManifest:
    _check_supported(cfg)

    index = gold_index if gold_index is not None else build_corpus_gold_index()
    retrieve = retrieve_fn if retrieve_fn is not None else _default_retrieve_fn(cfg)

    from rag_eval.providers.base import model_slug

    collection_names: dict[str, str] = {
        source: f"fastapi_{source}__{model_slug(cfg.embedding.model)}"
        for source in cfg.corpus.sources
    }

    metrics_by_dataset: dict[str, dict] = {}
    dataset_shas: dict[str, str] = {}
    retrieval_rows: list[dict] = []
    started = time.perf_counter()

    for dataset_name in cfg.eval.datasets:
        # verified == "no" means a human reviewer looked at this item and
        # rejected it (bad question, wrong gold, etc.) -- scoring against a
        # label that's known to be wrong would corrupt the metric, so it's
        # excluded here rather than left to silently drag the number down.
        # Unreviewed items (verified is None) are scored provisionally.
        raw_items = [item for item in load_dataset(dataset_name, eval_sets_dir) if item.verified != "no"]
        dataset_shas[dataset_name] = dataset_sha256(dataset_name, eval_sets_dir)

        per_item_for_agg: list[tuple[Sequence[str], set[str]]] = []
        for raw_item in raw_items:
            item = resolve_gold_chunks(raw_item, index)
            deny_ids = set(item.exclude_chunk_ids) if cfg.eval.self_retrieval == "holdout" else set()
            candidates = retrieve(item.question, cfg, deny_ids)
            item_metrics = _score_item(item, candidates, cfg.eval.k_values)

            retrieval_rows.append(
                {
                    "item_id": f"{dataset_name}::{item.id}",
                    "dataset": dataset_name,
                    "question": item.question,
                    "gold_chunk_ids": item.gold_chunk_ids,
                    "gold_granularity": item.gold_granularity,
                    "retrieved": [
                        {"chunk_id": c["chunk_id"], "url": c["url"], "score": c["score"], "rank": rank}
                        for rank, c in enumerate(candidates, start=1)
                    ],
                    "metrics": item_metrics,
                }
            )
            per_item_for_agg.append(([c["chunk_id"] for c in candidates], set(item.gold_chunk_ids)))

        metrics_by_dataset[dataset_name] = aggregate(per_item_for_agg, cfg.eval.k_values).to_dict()

    elapsed = time.perf_counter() - started

    run_dir = new_run(cfg, config_path, runs_root=runs_root)
    with open(run_dir / "retrieval.jsonl", "w", encoding="utf-8") as f:
        import json

        f.writelines(json.dumps(row) + "\n" for row in retrieval_rows)

    manifest = write_manifest(
        run_dir,
        cfg,
        config_path,
        metrics=metrics_by_dataset,
        timings={"total_seconds": elapsed, "items": len(retrieval_rows)},
        corpus_sha=corpus_sha if corpus_sha is not None else _combined_corpus_sha(),
        collection_names=collection_names,
        dataset_shas=dataset_shas,
    )
    return manifest
