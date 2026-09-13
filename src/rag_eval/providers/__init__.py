"""Provider registry: swappable LLM + embedding backends.

Provider/model selection here is explicit function arguments plus
module-level DEFAULT_* constants, deliberately not new `Settings` fields.
Phase 4's `RunConfig` (see docs/plan.md C1) is where these become
metric-affecting yaml config; until then a constant is the honest
placeholder, not a config system pretending to exist.
"""

from functools import lru_cache

from rag_eval.providers.base import EmbeddingProvider, LLMProvider

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = "fdm-llama"

DEFAULT_EMBEDDING_PROVIDER = "fastembed"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache
def get_llm(provider: str = DEFAULT_LLM_PROVIDER, model: str = DEFAULT_LLM_MODEL) -> LLMProvider:
    if provider == "ollama":
        from rag_eval.providers.llm.ollama import OllamaLLM

        return OllamaLLM(model=model)
    if provider == "groq":
        from rag_eval.providers.llm.groq import GroqLLM

        return GroqLLM(model=model)
    if provider == "gemini":
        from rag_eval.providers.llm.gemini import GeminiLLM

        return GeminiLLM(model=model)
    raise ValueError(f"Unknown LLM provider {provider!r}, expected 'ollama', 'groq', or 'gemini'")


@lru_cache
def get_embedder(
    provider: str = DEFAULT_EMBEDDING_PROVIDER, model: str = DEFAULT_EMBEDDING_MODEL
) -> EmbeddingProvider:
    if provider == "fastembed":
        from rag_eval.providers.embeddings.fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(model=model)
    if provider == "ollama":
        from rag_eval.providers.embeddings.ollama import OllamaEmbedder

        return OllamaEmbedder(model=model)
    raise ValueError(f"Unknown embedding provider {provider!r}, expected 'fastembed' or 'ollama'")


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_PROVIDER",
    "EmbeddingProvider",
    "LLMProvider",
    "get_embedder",
    "get_llm",
]
