"""Ollama embeddings via plain httpx -- kept as a local-dev backend and to
keep the nomic-embed-text collections queryable for the nomic-vs-bge A/B."""

from __future__ import annotations

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import model_slug

_KNOWN_DIMS = {"nomic-embed-text": 768}


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", base_url: str | None = None) -> None:
        self.model = model
        self.slug = model_slug(model)
        self.dim = _KNOWN_DIMS.get(model, 768)
        self._base_url = base_url or settings.ollama_base_url

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]
