import pytest

from rag_eval.rag.groundedness import groundedness_score, sentence_support, should_abstain
from rag_eval.retrieval.base import Candidate


def _cand(chunk_id: str, content: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, content=content, url=f"https://x/{chunk_id}", title="t", source_type="docs")


def test_sentence_support_returns_one_entry_per_sentence(fake_embedder):
    answer = "First sentence here. Second sentence here."
    candidates = [_cand("a", "First sentence here.")]

    supports = sentence_support(answer, candidates, fake_embedder)

    assert [s.sentence for s in supports] == ["First sentence here.", "Second sentence here."]


def test_sentence_matching_a_candidate_verbatim_scores_near_one(fake_embedder):
    # FakeEmbedder is deterministic: identical text -> identical vector ->
    # cosine similarity of exactly 1.0 against itself.
    candidates = [_cand("a", "FastAPI validates request bodies with Pydantic.")]
    answer = "FastAPI validates request bodies with Pydantic."

    supports = sentence_support(answer, candidates, fake_embedder)

    assert supports[0].score == pytest.approx(1.0)


def test_sentence_support_is_zero_when_no_candidates(fake_embedder):
    supports = sentence_support("Some answer sentence.", [], fake_embedder)
    assert supports[0].score == 0.0


def test_groundedness_score_is_mean_of_sentence_scores():
    from rag_eval.rag.groundedness import SentenceSupport

    supports = [SentenceSupport("s1", 1.0), SentenceSupport("s2", 0.0)]
    assert groundedness_score(supports) == 0.5


def test_groundedness_score_of_empty_supports_is_zero():
    assert groundedness_score([]) == 0.0


def test_should_abstain_below_threshold():
    assert should_abstain(0.3, threshold=0.45) is True
    assert should_abstain(0.6, threshold=0.45) is False
