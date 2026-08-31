"""Answer generation via the LLM provider layer, grounded in retrieved chunks.

`generate_answer` is the pre-Phase-7 uncited path: it still backs
rag/pipeline.py and the legacy pipeline-coupled RAGAS runner
(eval/run_ragas.py -- the artifact-driven judge is eval/judge.py as of
Phase 8, but run_ragas.py stays as the local pipeline+score convenience),
so it's kept exactly as it was rather than adapted -- neither of those
callers has a gold chunk_id list or an embedder to score groundedness
against.

`generate_cited_answer` is the Phase 7 path api/routes/ask.py uses: a
versioned, citation-forcing PromptTemplate against retrieval's own
`Candidate` shape, scored for citation validity and embedding groundedness.

`score_answer` is that scoring step pulled out on its own so
api/routes/ask.py's streaming route can call it after assembling a
streamed answer from `astream()` deltas -- sentence-level groundedness
needs the whole answer, so it can only run once the stream is done, but it
needs the exact same citation/groundedness/abstention logic as the
non-streaming path.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.common.schemas import RetrievedChunk
from rag_eval.providers import get_llm
from rag_eval.providers.base import EmbeddingProvider, LLMProvider
from rag_eval.rag.citations import CitationReport, validate_citations
from rag_eval.rag.groundedness import groundedness_score, sentence_support, should_abstain
from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.retrieval.base import Candidate

SYSTEM_PROMPT = (
    "You answer FastAPI questions using only the provided context, which is "
    "drawn from the FastAPI docs and GitHub Discussions. If the context does "
    "not contain the answer, say you don't know rather than guessing."
)

INSUFFICIENT_CONTEXT_PREFIX = "INSUFFICIENT_CONTEXT"


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n---\n\n".join(c.content for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(
    question: str, chunks: list[RetrievedChunk], llm: LLMProvider | None = None
) -> str:
    llm = llm or get_llm()
    response = llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ]
    )
    return response.content


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
