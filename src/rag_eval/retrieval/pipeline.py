"""RetrievalPipeline: wires the stages enabled by `RunConfig.retrieval` into
one `retrieve()` call.

`from_config` builds against real Chroma collections, fastembed, and LLM
providers. Direct construction takes each stage as an already-built object
(or `None` to disable it), so tests can inject fakes and assert a disabled
stage was never called (CLAUDE.md: prefer constructor injection over
patching) -- e.g. `RetrievalPipeline(dense=FakeDense(), bm25=None, ...)`.
"""

from __future__ import annotations

import time
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_eval.retrieval.base import Candidate, RetrievalResult
from rag_eval.retrieval.bm25 import Bm25Searcher
from rag_eval.retrieval.dense import DenseSearcher
from rag_eval.retrieval.expand import ParentExpander
from rag_eval.retrieval.fusion import apply_per_source_caps, reciprocal_rank_fusion
from rag_eval.retrieval.rerank import Reranker
from rag_eval.retrieval.rewrite import QueryRewriter

if TYPE_CHECKING:
    from rag_eval.config.run_config import RunConfig


@dataclass
class RetrievalPipeline:
    top_k: int
    candidates_k: int
    dense: DenseSearcher | None = None
    bm25: Bm25Searcher | None = None
    rewriter: QueryRewriter | None = None
    rewrite_n: int = 1
    reranker: Reranker | None = None
    rerank_top_n: int = 5
    expander: ParentExpander | None = None
    fusion_rrf_k: int = 60
    per_source_caps: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: RunConfig) -> RetrievalPipeline:
        from rag_eval.providers import get_embedder, get_llm
        from rag_eval.rag.vector_store import get_collection
        from rag_eval.rag.vector_store import query as vector_query
        from rag_eval.retrieval.bm25 import get_bm25_index
        from rag_eval.retrieval.rerank import FastEmbedReranker
        from rag_eval.retrieval.rewrite import HydeRewriter

        r = cfg.retrieval
        if not r.dense.enabled and not r.bm25.enabled:
            raise ValueError(
                "retrieval.dense.enabled and retrieval.bm25.enabled are both false "
                "-- at least one retrieval stage must be on, there is nothing to fuse otherwise"
            )

        embedder = get_embedder(cfg.embedding.provider, cfg.embedding.model)
        # Widened to list[str] (a fresh list, so list-invariance is a non-issue):
        # cfg.corpus.sources is list[Literal["docs", "discussions"]], but every
        # stage below is written against plain str source names.
        sources: list[str] = list(cfg.corpus.sources)

        dense = (
            DenseSearcher(sources=sources, embedder=embedder, query_fn=vector_query)
            if r.dense.enabled
            else None
        )

        bm25 = None
        if r.bm25.enabled:
            indices = {
                source: get_bm25_index(source, embedder, k1=r.bm25.k1, b=r.bm25.b)
                for source in sources
            }
            bm25 = Bm25Searcher(indices=indices)

        rewriter = None
        if r.query_rewrite.enabled:
            if r.query_rewrite.mode != "hyde":
                raise NotImplementedError(
                    f"retrieval.query_rewrite.mode={r.query_rewrite.mode!r} not implemented "
                    "(only 'hyde')"
                )
            llm = get_llm(cfg.generation.llm.provider, cfg.generation.llm.model)
            rewriter = HydeRewriter(llm=llm)

        reranker = FastEmbedReranker(model_name=r.rerank.model) if r.rerank.enabled else None

        expander = None
        if r.parent_expansion.enabled:
            if r.parent_expansion.mode != "section":
                raise NotImplementedError(
                    f"retrieval.parent_expansion.mode={r.parent_expansion.mode!r} not "
                    "implemented (only 'section')"
                )
            collections = {source: get_collection(source, embedder, create=False) for source in sources}
            expander = ParentExpander(collections=collections, max_tokens=r.parent_expansion.max_tokens)

        return cls(
            top_k=r.top_k,
            candidates_k=r.candidates_k,
            dense=dense,
            bm25=bm25,
            rewriter=rewriter,
            rewrite_n=r.query_rewrite.n,
            reranker=reranker,
            rerank_top_n=r.rerank.top_n,
            expander=expander,
            fusion_rrf_k=r.fusion.rrf_k,
            per_source_caps=r.per_source_caps.model_dump(),
        )

    def retrieve(
        self, query: str, *, k: int | None = None, deny_ids: AbstractSet[str] = frozenset()
    ) -> RetrievalResult:
        k = k if k is not None else self.top_k
        stage_timings: dict[str, float] = {}
        stage_counts: dict[str, int] = {}
        rewritten_queries: list[str] = []

        queries = [query]
        if self.rewriter is not None:
            t0 = time.perf_counter()
            rewritten_queries = self.rewriter.rewrite(query, self.rewrite_n)
            stage_timings["query_rewrite"] = time.perf_counter() - t0
            queries.extend(rewritten_queries)

        rankings: list[list[Candidate]] = []

        if self.dense is not None:
            t0 = time.perf_counter()
            dense_hits = 0
            for q in queries:
                ranking = self.dense.search(q, self.candidates_k, deny_ids)
                rankings.append(ranking)
                dense_hits += len(ranking)
            stage_timings["dense"] = time.perf_counter() - t0
            stage_counts["dense"] = dense_hits

        if self.bm25 is not None:
            t0 = time.perf_counter()
            bm25_ranking = self.bm25.search(query, self.candidates_k, deny_ids)
            rankings.append(bm25_ranking)
            stage_timings["bm25"] = time.perf_counter() - t0
            stage_counts["bm25"] = len(bm25_ranking)

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings, k=self.fusion_rrf_k)
        fused = apply_per_source_caps(fused, self.per_source_caps)
        stage_timings["fusion"] = time.perf_counter() - t0
        stage_counts["fusion"] = len(fused)

        candidates = fused[: self.candidates_k]

        if self.reranker is not None:
            t0 = time.perf_counter()
            candidates = self.reranker.rerank(query, candidates, self.rerank_top_n)
            stage_timings["rerank"] = time.perf_counter() - t0
            stage_counts["rerank"] = len(candidates)

        if self.expander is not None:
            t0 = time.perf_counter()
            candidates = self.expander.expand(candidates)
            stage_timings["expand"] = time.perf_counter() - t0

        return RetrievalResult(
            query=query,
            rewritten_queries=rewritten_queries,
            candidates=candidates[:k],
            stage_timings=stage_timings,
            stage_counts=stage_counts,
        )
