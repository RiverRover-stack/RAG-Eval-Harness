"""Query rewrite stage: HyDE (Hypothetical Document Embeddings).

Asks the LLM to write a plausible-sounding answer to the question, then
embeds and dense-searches *that* instead of (in addition to) the raw
question -- a hypothetical answer's embedding tends to land closer to real
answer chunks than a short question's embedding does. Costs one extra LLM
call per rewrite (`n` of them), so its latency has to be weighed against its
recall gain (docs/plan.md Phase 6: "Record HyDE's latency cost beside its
recall gain").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rag_eval.providers.base import LLMProvider

_HYDE_PROMPT = (
    "Write a short, plausible answer to the following question about FastAPI. "
    "It does not need to be factually correct -- it only needs to read like a "
    "real answer, using the vocabulary a real answer would use.\n\n"
    "Question: {question}\n\nAnswer:"
)


class QueryRewriter(Protocol):
    def rewrite(self, query: str, n: int) -> list[str]: ...


@dataclass
class HydeRewriter:
    llm: LLMProvider
    temperature: float = 0.7
    max_tokens: int = 256

    def rewrite(self, query: str, n: int) -> list[str]:
        messages = [{"role": "user", "content": _HYDE_PROMPT.format(question=query)}]
        rewrites = []
        for _ in range(n):
            response = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
            rewrites.append(response.content.strip())
        return rewrites
