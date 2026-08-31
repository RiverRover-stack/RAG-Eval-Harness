"""Provider protocols shared by every LLM/embedding backend.

`astream` lands here in Phase 7 (docs/plan.md) now that SSE gives it a real
consumer (api/routes/ask.py's /api/ask/stream). Implemented for Groq (SSE)
and Ollama (NDJSON, not SSE -- a real framing difference, not a typo) since
those are the serving default and the local-dev backend; Gemini and
AirForce raise NotImplementedError rather than silently falling back to
non-streaming, since a caller asking a fallback-only provider to stream
should find out loudly, not get a single big chunk pretending to be one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class StreamChunk:
    delta: str
    done: bool = False
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

    def astream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]: ...


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
