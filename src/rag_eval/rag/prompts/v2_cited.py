"""v2 prompt: numbered, citation-forcing context blocks.

Each candidate becomes context block `[n]` tagged with its source URL; the
model must cite `[n]` immediately before the closing punctuation of every
sentence that uses it (a sentence may carry more than one citation, e.g.
`[1][2]`), and must emit the INSUFFICIENT_CONTEXT sentinel instead of
guessing when the context doesn't answer the question. The sentinel exists
because an honest "I don't know" under the old prompt scored 0 on every
RAGAS metric (docs/plan.md) -- abstention needs to be a distinguishable
outcome, not a total-failure signal.
"""

from __future__ import annotations

from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.retrieval.base import Candidate

SYSTEM_PROMPT = (
    "You answer FastAPI questions using only the numbered context blocks "
    "below, drawn from the FastAPI docs and GitHub Discussions. Cite the "
    "block number in square brackets, e.g. [1], immediately before the "
    "closing punctuation of every sentence that relies on it -- a sentence "
    "may carry more than one citation, e.g. [1][2]. Never cite a number "
    "that has no context block. If the context blocks do not contain "
    "enough information to answer the question, respond with exactly "
    "'INSUFFICIENT_CONTEXT: <what is missing>' and nothing else."
)


def build_user_prompt(question: str, candidates: list[Candidate]) -> str:
    blocks = "\n\n".join(
        f"[{i}] (source: {c.url})\n{c.content}" for i, c in enumerate(candidates, start=1)
    )
    return f"{blocks}\n\nQuestion: {question}\n\nAnswer:"


PROMPT = PromptTemplate(
    version="v2-cited", system_prompt=SYSTEM_PROMPT, build_user_prompt=build_user_prompt
)
