"""Answer generation via the LLM provider layer, grounded in retrieved chunks.

`generate_cited_answer` is the one generation path in the codebase: a
versioned, citation-forcing PromptTemplate against retrieval's own
`Candidate` shape, scored for citation validity and embedding groundedness.
It backs both api/routes/ask.py (the live API) and eval/generate.py (Phase
8's batch generation for judging).

`score_answer` is that scoring step pulled out on its own so
api/routes/ask.py's streaming route can call it after assembling a
streamed answer from `astream()` deltas -- sentence-level groundedness
needs the whole answer, so it can only run once the stream is done, but it
needs the exact same citation/groundedness/abstention logic as the
non-streaming path.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.providers.base import EmbeddingProvider, LLMProvider
from rag_eval.rag.citations import CitationReport, validate_citations
from rag_eval.rag.groundedness import groundedness_score, sentence_support, should_abstain
from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.retrieval.base import Candidate

INSUFFICIENT_CONTEXT_PREFIX = "INSUFFICIENT_CONTEXT"


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    citations: CitationReport
    groundedness: float
    abstained: bool
    prompt_tokens: int | None
    completion_tokens: int | None


def score_answer(
    answer: str,
    candidates: list[Candidate],
    embedder: EmbeddingProvider,
    abstain_below: float,
) -> tuple[CitationReport, float, bool]:
    """(citations, groundedness, abstained). An INSUFFICIENT_CONTEXT answer
    short-circuits straight to `abstained=True` -- it's already an honest
    abstention, not something citation/groundedness scoring should
    second-guess. Otherwise abstention is decided post-hoc: a
    confidently-worded answer with low embedding support against what was
    actually retrieved abstains anyway."""
    if answer.strip().startswith(INSUFFICIENT_CONTEXT_PREFIX):
        return (
            CitationReport(citations=[], unknown_indices=[], uncited_sentences=[], coverage=0.0),
            0.0,
            True,
        )
    citations = validate_citations(answer, candidates)
    score = groundedness_score(sentence_support(answer, candidates, embedder))
    return citations, score, should_abstain(score, abstain_below)


def generate_cited_answer(
    question: str,
    candidates: list[Candidate],
    *,
    prompt: PromptTemplate,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    abstain_below: float = 0.45,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> GenerationResult:
    """Build `prompt` against `candidates`, call the LLM, and score the result."""
    response = llm.complete(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.build_user_prompt(question, candidates)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = response.content
    citations, score, abstained = score_answer(answer, candidates, embedder, abstain_below)
    return GenerationResult(
        answer=answer,
        citations=citations,
        groundedness=score,
        abstained=abstained,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )
