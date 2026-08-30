from rag_eval.rag.citations import StreamingCitationScanner, extract_citations, validate_citations
from rag_eval.retrieval.base import Candidate


def _cand(chunk_id: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, content=f"content-{chunk_id}", url=f"https://x/{chunk_id}", title="t", source_type="docs")


CANDIDATES = [_cand("a"), _cand("b"), _cand("c")]


def test_extract_citations_reports_offsets():
    answer = "FastAPI is fast [1]."
    citations = extract_citations(answer, CANDIDATES)

    assert len(citations) == 1
    c = citations[0]
    assert c.index == 1
    assert c.chunk_id == "a"
    assert c.url == "https://x/a"
    assert answer[c.char_start : c.char_end] == "[1]"


def test_extract_citations_handles_adjacent_markers():
    answer = "FastAPI supports both sync and async [1][2]."
    citations = extract_citations(answer, CANDIDATES)

    assert [c.index for c in citations] == [1, 2]
    assert [c.chunk_id for c in citations] == ["a", "b"]


def test_extract_citations_ignores_index_with_no_candidate():
    answer = "This cites a chunk that doesn't exist [7]."
    assert extract_citations(answer, CANDIDATES) == []


def test_validate_citations_reports_unknown_indices():
    answer = "Known [1]. Unknown [7]."
    report = validate_citations(answer, CANDIDATES)

    assert report.unknown_indices == [7]
    assert [c.index for c in report.citations] == [1]


def test_validate_citations_flags_uncited_sentences_and_computes_coverage():
    answer = "This sentence is cited [1]. This one is not."
    report = validate_citations(answer, CANDIDATES)

    assert report.uncited_sentences == ["This one is not."]
    assert report.coverage == 0.5


def test_validate_citations_full_coverage_when_every_sentence_cited():
    answer = "First point [1]. Second point [2]."
    report = validate_citations(answer, CANDIDATES)

    assert report.uncited_sentences == []
    assert report.coverage == 1.0


def test_validate_citations_on_insufficient_context_sentinel_has_no_citations():
    answer = "INSUFFICIENT_CONTEXT: the docs don't cover rate limiting"
    report = validate_citations(answer, CANDIDATES)

    assert report.citations == []
    assert report.unknown_indices == []


def test_streaming_scanner_reassembles_a_marker_split_across_feeds():
    scanner = StreamingCitationScanner(CANDIDATES)

    assert scanner.feed("FastAPI is fast [") == []
    citations = scanner.feed("1].")

    assert [c.index for c in citations] == [1]
    assert [c.chunk_id for c in citations] == ["a"]


def test_streaming_scanner_never_emits_the_same_citation_twice():
    scanner = StreamingCitationScanner(CANDIDATES)

    first = scanner.feed("Cited [1]. ")
    second = scanner.feed("More text, no new citation here.")
    third = scanner.feed(" Another [2].")

    assert [c.index for c in first] == [1]
    assert second == []
    assert [c.index for c in third] == [2]


def test_streaming_scanner_handles_arbitrary_chunk_boundaries():
    text = "Sync and async both work [1][2]."
    scanner = StreamingCitationScanner(CANDIDATES)

    all_citations = []
    for ch in text:  # feed one character at a time -- the extreme split case
        all_citations.extend(scanner.feed(ch))

    assert [c.index for c in all_citations] == [1, 2]


def test_streaming_scanner_ignores_unknown_index():
    scanner = StreamingCitationScanner(CANDIDATES)
    assert scanner.feed("Cites nothing real [9].") == []
