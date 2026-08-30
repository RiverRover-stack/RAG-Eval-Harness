from rag_eval.providers.base import LLMResponse
from rag_eval.rag.generator import generate_cited_answer
from rag_eval.rag.prompts import PROMPTS
from rag_eval.retrieval.base import Candidate


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self._content = content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)
        return LLMResponse(
            content=self._content,
            model=self.model,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )


def _cand(chunk_id: str, content: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, content=content, url=f"https://x/{chunk_id}", title="t", source_type="docs")


CANDIDATES = [_cand("a", "FastAPI validates request bodies with Pydantic models.")]


def test_generate_cited_answer_calls_llm_with_v2_cited_prompt(fake_embedder):
    llm = FakeLLM("FastAPI validates bodies with Pydantic [1].")
    generate_cited_answer(
        "How does FastAPI validate request bodies?",
        CANDIDATES,
        prompt=PROMPTS["v2-cited"],
        llm=llm,
        embedder=fake_embedder,
    )

    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == PROMPTS["v2-cited"].system_prompt
    assert "[1] (source: https://x/a)" in messages[1]["content"]


def test_generate_cited_answer_reports_citations(fake_embedder):
    llm = FakeLLM("FastAPI validates bodies with Pydantic [1].")
    result = generate_cited_answer(
        "q", CANDIDATES, prompt=PROMPTS["v2-cited"], llm=llm, embedder=fake_embedder
    )

    assert [c.index for c in result.citations.citations] == [1]
    assert result.abstained is False
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_generate_cited_answer_insufficient_context_short_circuits_to_abstained(fake_embedder):
    llm = FakeLLM("INSUFFICIENT_CONTEXT: rate limiting isn't covered by the docs")
    result = generate_cited_answer(
        "How do I rate-limit an endpoint?",
        CANDIDATES,
        prompt=PROMPTS["v2-cited"],
        llm=llm,
        embedder=fake_embedder,
    )

    assert result.abstained is True
    assert result.citations.citations == []
    assert result.groundedness == 0.0


def test_generate_cited_answer_abstains_on_low_groundedness_even_without_sentinel(fake_embedder):
    # An answer with no overlap at all with the retrieved content's
    # (deterministic, hashed) embedding should score low support and
    # abstain post-hoc, even though it never says INSUFFICIENT_CONTEXT.
    llm = FakeLLM("Something completely unrelated to the context [1].")
    result = generate_cited_answer(
        "q",
        CANDIDATES,
        prompt=PROMPTS["v2-cited"],
        llm=llm,
        embedder=fake_embedder,
        abstain_below=0.99,  # force the low-groundedness path deterministically
    )

    assert result.abstained is True


def test_generate_cited_answer_does_not_abstain_above_threshold(fake_embedder):
    llm = FakeLLM("FastAPI validates request bodies with Pydantic models [1].")
    result = generate_cited_answer(
        "q",
        CANDIDATES,
        prompt=PROMPTS["v2-cited"],
        llm=llm,
        embedder=fake_embedder,
        abstain_below=-1.0,  # cosine similarity is never below -1
    )

    assert result.abstained is False
