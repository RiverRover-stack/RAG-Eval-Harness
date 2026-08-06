"""In-process ONNX embeddings via fastembed -- no torch, no server.

See docs/adr/0002-fastembed-over-hosted-embeddings.md for why this replaced
a hosted embeddings API.
"""

from __future__ import annotations

from rag_eval.common.config import settings
from rag_eval.providers.base import model_slug

# fastembed's own registry carries each model's output dim; hardcoded here
# for the one model this project uses so `.dim` doesn't require loading the
# model first.
_KNOWN_DIMS = {"BAAI/bge-small-en-v1.5": 384}


class FastEmbedEmbedder:
    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = model
        self.slug = model_slug(model)
        self.dim = _KNOWN_DIMS.get(model, 384)
        self._model = None  # lazy: avoid loading the ONNX model until first use

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model, cache_dir=settings.fastembed_cache_dir
            )
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [vec.tolist() for vec in self._get_model().embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
