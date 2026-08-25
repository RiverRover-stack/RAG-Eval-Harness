"""Extract and validate `[n]`-style citations against the candidates a
v2-cited answer was generated from (docs/plan.md Phase 7).

The v2-cited prompt instructs the model to place a citation marker
immediately before a sentence's closing punctuation, so a citation-aware
sentence split can attribute each marker to the sentence it supports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_eval.retrieval.base import Candidate

_CITATION_RE = re.compile(r"\[(\d+)\]")
# A "sentence" runs up to and including its terminal punctuation, or -- for
# a trailing fragment with none, e.g. the INSUFFICIENT_CONTEXT sentinel --
# to the end of the text.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    url: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class CitationReport:
    citations: list[Citation]
    unknown_indices: list[int]
    uncited_sentences: list[str]
    coverage: float


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """(sentence, start, end) triples, `end` trimmed to the last non-space
    character so a citation marker placed before the terminal punctuation
    is still inside the span."""
    spans = []
    for m in _SENTENCE_RE.finditer(text):
        stripped = m.group().rstrip()
        if not stripped.strip():
            continue
        start = m.start()
        spans.append((stripped.strip(), start, start + len(stripped)))
    return spans


def _find_markers(answer: str) -> list[tuple[int, int, int]]:
    return [(int(m.group(1)), m.start(), m.end()) for m in _CITATION_RE.finditer(answer)]


def extract_citations(answer: str, candidates: list[Candidate]) -> list[Citation]:
    citations = []
    for index, start, end in _find_markers(answer):
        if 1 <= index <= len(candidates):
            c = candidates[index - 1]
            citations.append(
                Citation(index=index, chunk_id=c.chunk_id, url=c.url, char_start=start, char_end=end)
            )
    return citations


def validate_citations(answer: str, candidates: list[Candidate]) -> CitationReport:
    markers = _find_markers(answer)
    citations = extract_citations(answer, candidates)
    unknown_indices = sorted({idx for idx, _, _ in markers if not (1 <= idx <= len(candidates))})

    sentences = split_sentences(answer)
    uncited_sentences = []
    cited_count = 0
    for sentence, s_start, s_end in sentences:
        if any(s_start <= start < s_end for _, start, _ in markers):
            cited_count += 1
        else:
            uncited_sentences.append(sentence)
    coverage = cited_count / len(sentences) if sentences else 1.0

    return CitationReport(
        citations=citations,
        unknown_indices=unknown_indices,
        uncited_sentences=uncited_sentences,
        coverage=coverage,
    )


class StreamingCitationScanner:
    """Incremental citation extraction for SSE token streams.

    A `[n]` marker can arrive split across two `feed()` calls, so only the
    buffer up to the last `]` seen so far is scanned -- anything after that
    might be a marker still in flight.
    """

    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = candidates
        self._buffer = ""
        self._scanned_to = 0

    def feed(self, delta: str) -> list[Citation]:
        self._buffer += delta
        safe_end = self._buffer.rfind("]") + 1
        if safe_end <= self._scanned_to:
            return []

        region = self._buffer[self._scanned_to : safe_end]
        offset = self._scanned_to
        self._scanned_to = safe_end

        new_citations = []
        for m in _CITATION_RE.finditer(region):
            index = int(m.group(1))
            if 1 <= index <= len(self._candidates):
                c = self._candidates[index - 1]
                new_citations.append(
                    Citation(
                        index=index,
                        chunk_id=c.chunk_id,
                        url=c.url,
                        char_start=offset + m.start(),
                        char_end=offset + m.end(),
                    )
                )
        return new_citations
