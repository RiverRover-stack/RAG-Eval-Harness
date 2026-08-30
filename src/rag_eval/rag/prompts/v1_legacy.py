"""v1 prompt: the original un-cited generator prompt (rag/generator.py,
pre-Phase-7), kept as a selectable PromptTemplate rather than deleted, so a
`none`-citation run stays comparable to the historical baseline.
"""

from __future__ import annotations

from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.retrieval.base import Candidate

SYSTEM_PROMPT = (
    "You answer FastAPI questions using only the provided context, which is "
    "drawn from the FastAPI docs and GitHub Discussions. If the context does "
    "not contain the answer, say you don't know rather than guessing."
)


def build_user_prompt(question: str, candidates: list[Candidate]) -> str:
    context = "\n\n---\n\n".join(c.content for c in candidates)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


PROMPT = PromptTemplate(
    version="v1-legacy", system_prompt=SYSTEM_PROMPT, build_user_prompt=build_user_prompt
)
