"""BM25 lexical search stage.

Hand-rolled Okapi BM25 rather than a dependency: the corpus is small enough
(~1200 chunks, plan target < 1s) that a pure-Python inverted index is
simpler than adding a new package for it.

Tokenizer is code-aware: lowercase, split on non-alphanumeric, but keep `_`
and `.` *inside* identifiers so `response_model` and `jsonable_encoder`
survive as single terms instead of splitting into noise tokens -- on a
code-docs corpus that's the difference between BM25 helping and BM25 being
noise (docs/plan.md Phase 6).

`Bm25Index` is built once per collection from `collection.get(...)` and
cached in `_INDEX_CACHE`, keyed by the collection name, so repeated queries
against the same run don't re-scan the corpus.
"""

from __future__ import annotations

import math
import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_eval.retrieval.base import Candidate

if TYPE_CHECKING:
    from rag_eval.providers.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Bm25Index:
    doc_ids: list[str]
    doc_contents: list[str]
    doc_urls: list[str]
    doc_titles: list[str]
    doc_lengths: list[int]
    avg_doc_length: float
    n_docs: int
    inverted: dict[str, dict[int, int]]  # term -> {doc_idx: term_freq}
    doc_freq: dict[str, int]  # term -> number of docs containing it
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def build(
        cls,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> Bm25Index:
        inverted: dict[str, dict[int, int]] = {}
        doc_lengths: list[int] = []
        for idx, doc in enumerate(documents):
            term_freqs: dict[str, int] = {}
            for term in tokenize(doc):
                term_freqs[term] = term_freqs.get(term, 0) + 1
            doc_lengths.append(sum(term_freqs.values()))
            for term, freq in term_freqs.items():
                inverted.setdefault(term, {})[idx] = freq

        n_docs = len(documents)
        avg_doc_length = (sum(doc_lengths) / n_docs) if n_docs else 0.0
        doc_freq = {term: len(postings) for term, postings in inverted.items()}
        metas = [m or {} for m in metadatas]
        return cls(
            doc_ids=ids,
            doc_contents=documents,
            doc_urls=[m.get("url", "") for m in metas],
            doc_titles=[m.get("title", "") for m in metas],
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
            n_docs=n_docs,
            inverted=inverted,
            doc_freq=doc_freq,
            k1=k1,
            b=b,
        )

    def _idf(self, term: str) -> float:
        n_t = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def score_all(self, query_text: str, source_type: str) -> list[Candidate]:
        """Score every doc containing at least one query term. Unranked --
        the caller (Bm25Searcher) merges across sources and assigns rank.

        `source_type` is the searcher's source key ("docs"/"discussions"),
        not the chunk metadata's own source_type field -- see the note in
        Bm25Searcher.search."""
        if self.n_docs == 0 or self.avg_doc_length == 0:
            return []

        scores: dict[int, float] = {}
        for term in tokenize(query_text):
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_idx, freq in postings.items():
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[doc_idx] / self.avg_doc_length
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (freq * (self.k1 + 1)) / denom

        return [
            Candidate(
                chunk_id=self.doc_ids[i],
                content=self.doc_contents[i],
                url=self.doc_urls[i],
                title=self.doc_titles[i],
                source_type=source_type,
                scores={"bm25": score},
                stages=["bm25"],
                final_score=score,
            )
            for i, score in scores.items()
        ]


_INDEX_CACHE: dict[str, Bm25Index] = {}


def build_bm25_index_from_collection(collection, *, k1: float = 1.2, b: float = 0.75) -> Bm25Index:
    result = collection.get(include=["documents", "metadatas"])
    return Bm25Index.build(
        result["ids"] or [],
        result["documents"] or [],
        result["metadatas"] or [],
        k1=k1,
        b=b,
    )


def get_bm25_index(
    source: str,
    embedder: EmbeddingProvider,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> Bm25Index:
    """Module-cached by collection name -- built once per (source, embedder)
    pair from the live Chroma collection."""
    from rag_eval.rag.vector_store import collection_name, get_collection

    key = collection_name(source, embedder)
    if key not in _INDEX_CACHE:
        collection = get_collection(source, embedder, create=False)
        _INDEX_CACHE[key] = build_bm25_index_from_collection(collection, k1=k1, b=b)
    return _INDEX_CACHE[key]


@dataclass
class Bm25Searcher:
    indices: dict[str, Bm25Index] = field(default_factory=dict)

    def search(
        self, query_text: str, k: int, deny_ids: AbstractSet[str] = frozenset()
    ) -> list[Candidate]:
        # `indices` keys ("docs"/"discussions") are the searcher's source
        # names, which match cfg.corpus.sources and per_source_caps -- that's
        # what gets stamped onto each Candidate.source_type, not whatever the
        # chunk's own metadata.source_type field happens to say.
        hits: list[Candidate] = []
        for source, index in self.indices.items():
            hits.extend(index.score_all(query_text, source))
        hits = [c for c in hits if c.chunk_id not in deny_ids]
        hits.sort(key=lambda c: c.scores["bm25"], reverse=True)
        top = hits[:k]
        for rank, c in enumerate(top, start=1):
            c.ranks["bm25"] = rank
        return top
