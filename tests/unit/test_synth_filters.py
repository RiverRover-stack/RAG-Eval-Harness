"""Each auto-filter is proven on a planted positive (fires) and a planted
negative (stays quiet), independent of any real LLM or embedding model --
per CLAUDE.md, constructor injection over patching.
"""

import pytest

from rag_eval.eval.synth_eval_set import (
    BuildReport,
    build_docs_synth_v1,
    build_trigram_doc_freq,
    closed_book_reject,
    lexical_overlap_reject,
    near_duplicate_reject,
    stratified_sample,
)
from rag_eval.providers.base import LLMResponse


class ScriptedLLM:
    """Routes a canned response by matching a marker substring in the last
    user message; can also be told to blow up for a given marker."""

    name = "fake"
    model = "fake-llm"

    def __init__(self, by_marker: dict[str, str], raise_markers: frozenset[str] = frozenset()):
        self._by_marker = by_marker
        self._raise_markers = raise_markers
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)
        content = messages[-1]["content"]
        for marker in self._raise_markers:
            if marker in content:
                raise RuntimeError("simulated generation failure")
        for marker, response in self._by_marker.items():
            if marker in content:
                return LLMResponse(content=response, model=self.model)
        raise AssertionError(f"ScriptedLLM has no response for: {content[:120]!r}")


class VectorMapEmbedder:
    name = "fake"
    model = "fake-embed"
    dim = 3
    slug = "fake"

    def __init__(self, vectors: dict[str, list[float]], default: list[float] | None = None):
        self._vectors = vectors
        self._default = default or [0.0, 0.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors.get(text, list(self._default))

    def embed_documents(self, texts, batch_size=64):
        return [self.embed_query(t) for t in texts]


# ---------------------------------------------------------------------------
# lexical_overlap_reject
# ---------------------------------------------------------------------------

CHUNK_TEXT = "The zxqvy_flibbertigibbet_token controls internal retry jitter timing exactly."


def test_lexical_overlap_fires_on_a_verbatim_rare_trigram():
    df = build_trigram_doc_freq([CHUNK_TEXT])
    question = "Why does the zxqvy_flibbertigibbet_token controls internal retry jitter?"
    assert lexical_overlap_reject(question, CHUNK_TEXT, df) is True


def test_lexical_overlap_stays_quiet_on_a_paraphrase():
    df = build_trigram_doc_freq([CHUNK_TEXT])
    question = "What setting adjusts the delay between retry attempts?"
    assert lexical_overlap_reject(question, CHUNK_TEXT, df) is False


def test_lexical_overlap_ignores_common_trigrams():
    # a trigram common across many chunks isn't "rare" even if it's shared.
    corpus = [CHUNK_TEXT] + [f"you can use this for {i} things" for i in range(5)]
    df = build_trigram_doc_freq(corpus)
    question = "How do I use this for something?"
    assert lexical_overlap_reject(question, CHUNK_TEXT, df) is False


# ---------------------------------------------------------------------------
# closed_book_reject
# ---------------------------------------------------------------------------


def test_closed_book_fires_when_answer_already_matches_the_passage():
    llm = ScriptedLLM({"what does X do": "X controls the retry timing exactly."})
    embedder = VectorMapEmbedder(
        {
            "X controls the retry timing exactly.": [1.0, 0.0, 0.0],
            CHUNK_TEXT: [0.99, 0.01, 0.0],
        }
    )
    assert (
        closed_book_reject("what does X do", CHUNK_TEXT, closed_book_llm=llm, embedder=embedder)
        is True
    )


def test_closed_book_stays_quiet_when_model_has_no_idea():
    llm = ScriptedLLM({"what does X do": "I'm not sure, could be anything."})
    embedder = VectorMapEmbedder(
        {
            "I'm not sure, could be anything.": [1.0, 0.0, 0.0],
            CHUNK_TEXT: [0.0, 1.0, 0.0],
        }
    )
    assert (
        closed_book_reject("what does X do", CHUNK_TEXT, closed_book_llm=llm, embedder=embedder)
        is False
    )


# ---------------------------------------------------------------------------
# near_duplicate_reject
# ---------------------------------------------------------------------------


def test_near_duplicate_fires_on_a_close_vector():
    accepted = [[1.0, 0.0, 0.0]]
    assert near_duplicate_reject([0.99, 0.01, 0.0], accepted) is True


def test_near_duplicate_stays_quiet_on_an_orthogonal_vector():
    accepted = [[1.0, 0.0, 0.0]]
    assert near_duplicate_reject([0.0, 1.0, 0.0], accepted) is False


def test_near_duplicate_stays_quiet_with_no_accepted_items_yet():
    assert near_duplicate_reject([1.0, 0.0, 0.0], []) is False


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, section: str) -> dict:
    return {
        "id": chunk_id,
        "document": f"content for {chunk_id}",
        "metadata": {"title": chunk_id, "path": f"{section}/page.md", "url": f"https://x/{chunk_id}"},
    }


def test_stratified_sample_covers_every_section():
    chunks = (
        [_chunk(f"t{i}", "tutorial") for i in range(10)]
        + [_chunk(f"a{i}", "advanced") for i in range(2)]
        + [_chunk(f"r{i}", "reference") for i in range(1)]
    )
    sample = stratified_sample(chunks, n=3, seed=0)
    sections = {c["metadata"]["path"].split("/")[0] for c in sample}
    assert sections == {"tutorial", "advanced", "reference"}


def test_stratified_sample_deterministic_given_seed():
    chunks = [_chunk(f"t{i}", "tutorial") for i in range(10)]
    a = stratified_sample(chunks, n=5, seed=42)
    b = stratified_sample(chunks, n=5, seed=42)
    assert [c["id"] for c in a] == [c["id"] for c in b]


def test_stratified_sample_caps_at_available_chunks():
    chunks = [_chunk("t0", "tutorial")]
    assert len(stratified_sample(chunks, n=10, seed=0)) == 1


# ---------------------------------------------------------------------------
# build_docs_synth_v1 orchestration
# ---------------------------------------------------------------------------

CHUNK_TUTORIAL = {
    "id": "c-tutorial",
    "document": "FastAPI lets you declare response models using Pydantic classes for validation.",
    "metadata": {"title": "T1", "path": "tutorial/response-model.md", "url": "https://x/1"},
}
CHUNK_ADVANCED = {
    "id": "c-advanced",
    "document": CHUNK_TEXT,
    "metadata": {"title": "T2", "path": "advanced/retry.md", "url": "https://x/2"},
}


def test_build_docs_synth_v1_rejects_lexical_and_retains_clean_item():
    gen_llm = ScriptedLLM(
        {
            "Page: T1": "How do you validate API responses in FastAPI?",
            "Page: T2": "Why does the zxqvy_flibbertigibbet_token controls internal retry jitter?",
        }
    )
    closed_book_llm = ScriptedLLM({"How do you validate": "not sure, maybe logging?"})
    embedder = VectorMapEmbedder(
        {
            CHUNK_TUTORIAL["document"]: [1.0, 0.0, 0.0],
            "not sure, maybe logging?": [0.0, 1.0, 0.0],
            "How do you validate API responses in FastAPI?": [0.0, 0.0, 1.0],
        }
    )

    items, report = build_docs_synth_v1(
        [CHUNK_TUTORIAL, CHUNK_ADVANCED],
        llm=gen_llm,
        closed_book_llm=closed_book_llm,
        embedder=embedder,
        n_target=2,
        seed=0,
    )

    assert report.generated == 2
    assert report.rejected_lexical == 1
    assert report.retained == 1
    assert report.rejected_closed_book == 0
    assert report.sections_covered == {"tutorial"}
    assert len(items) == 1
    assert items[0].id == "c-tutorial"
    assert items[0].gold_urls == ["https://x/1"]
    assert items[0].dataset == "docs_synth_v1"


def test_build_docs_synth_v1_calls_checkpoint_fn_once_per_chunk_regardless_of_outcome():
    gen_llm = ScriptedLLM(
        {
            "Page: T1": "How do you validate API responses in FastAPI?",
            "Page: T2": "Why does the zxqvy_flibbertigibbet_token controls internal retry jitter?",
        }
    )
    closed_book_llm = ScriptedLLM({"How do you validate": "not sure, maybe logging?"})
    embedder = VectorMapEmbedder(
        {
            CHUNK_TUTORIAL["document"]: [1.0, 0.0, 0.0],
            "not sure, maybe logging?": [0.0, 1.0, 0.0],
            "How do you validate API responses in FastAPI?": [0.0, 0.0, 1.0],
        }
    )
    checkpoints: list[tuple[int, int]] = []

    def checkpoint_fn(items, report):
        checkpoints.append((len(items), report.generated))

    build_docs_synth_v1(
        [CHUNK_TUTORIAL, CHUNK_ADVANCED],
        llm=gen_llm,
        closed_book_llm=closed_book_llm,
        embedder=embedder,
        n_target=2,
        seed=0,
        checkpoint_fn=checkpoint_fn,
    )
    # one checkpoint per sampled chunk, in order -- the rejected one first
    # (advanced, alphabetically before tutorial in stratified_sample),
    # then the retained one.
    assert len(checkpoints) == 2
    assert checkpoints[-1][0] == 1  # final checkpoint sees the one retained item


def test_build_docs_synth_v1_counts_generation_errors():
    gen_llm = ScriptedLLM({}, raise_markers=frozenset({"Page: T1", "Page: T2"}))
    closed_book_llm = ScriptedLLM({})
    embedder = VectorMapEmbedder({})

    items, report = build_docs_synth_v1(
        [CHUNK_TUTORIAL, CHUNK_ADVANCED],
        llm=gen_llm,
        closed_book_llm=closed_book_llm,
        embedder=embedder,
        n_target=2,
        seed=0,
    )
    assert items == []
    assert report.generated == 0
    assert report.rejected_generation_error == 2


def test_build_docs_synth_v1_rejects_empty_question():
    gen_llm = ScriptedLLM({"Page: T1": "", "Page: T2": ""})
    closed_book_llm = ScriptedLLM({})
    embedder = VectorMapEmbedder({})

    items, report = build_docs_synth_v1(
        [CHUNK_TUTORIAL],
        llm=gen_llm,
        closed_book_llm=closed_book_llm,
        embedder=embedder,
        n_target=1,
        seed=0,
    )
    assert items == []
    assert report.rejected_generation_error == 1


def test_build_report_summary_reads_like_the_readme_sentence():
    report = BuildReport(
        generated=150,
        rejected_lexical=9,
        rejected_closed_book=11,
        rejected_duplicate=3,
        retained=127,
    )
    summary = report.summary()
    assert "150 generated" in summary
    assert "127 retained" in summary
    assert "9 lexical" in summary
    assert "11 closed-book" in summary
    assert "3 dup" in summary


@pytest.mark.parametrize("field", ["rejected_lexical", "rejected_closed_book", "rejected_duplicate"])
def test_build_report_to_dict_round_trips(field):
    report = BuildReport(**{field: 5})
    assert report.to_dict()[field] == 5
