"""Provider protocols shared by every LLM/embedding backend.

`LLMProvider` implements `complete()` only for now -- `astream` is added in
Phase 7 (docs/plan.md) once SSE gives it a real consumer and
`test_ask_stream.py` gives it a test. Adding it earlier would mean ~60 lines
of streaming code with nothing exercising it for two phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int
    slug: str  # e.g. "bge-small-en-v15"

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def model_slug(model: str) -> str:
    """'BAAI/bge-small-en-v1.5' -> 'bge-small-en-v15'; 'nomic-embed-text' unchanged.

    Strips any org prefix and drops '.' so the result is a safe Chroma
    collection-name suffix (docs/plan.md: "fastapi_docs__bge-small-en-v15").
    """
    return model.split("/")[-1].replace(".", "").lower()
