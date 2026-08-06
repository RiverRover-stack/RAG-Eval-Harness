"""Answer generation via the LLM provider layer, grounded in retrieved chunks.

Defaults to Ollama (providers.DEFAULT_LLM_PROVIDER) until Phase 4's
RunConfig makes the serving LLM a yaml value -- switching the default to
Groq here would confound any nomic-vs-bge retrieval comparison with a
generator change.
"""

from rag_eval.common.schemas import RetrievedChunk
from rag_eval.providers import get_llm
from rag_eval.providers.base import LLMProvider

SYSTEM_PROMPT = (
    "You answer FastAPI questions using only the provided context, which is "
    "drawn from the FastAPI docs and GitHub Discussions. If the context does "
    "not contain the answer, say you don't know rather than guessing."
)


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
