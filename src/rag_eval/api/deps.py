"""FastAPI dependency wiring for the Phase 7 app factory (docs/plan.md).

`build_app_state` does the heavy lifting once, at lifespan startup --
loading the `RunConfig` and building the `RetrievalPipeline` from it (which
in turn warms the embedder, reranker, and BM25 index), plus resolving the
serving LLM and prompt template -- so no request ever pays for any of that.
`get_app_state` is the per-request dependency every route pulls it back
through.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request

from rag_eval.config.run_config import RunConfig, load_run_config
from rag_eval.providers import get_embedder, get_llm
from rag_eval.providers.base import EmbeddingProvider, LLMProvider
from rag_eval.rag.prompts import PROMPTS
from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.retrieval.pipeline import RetrievalPipeline


@dataclass
class AppState:
    cfg: RunConfig
    pipeline: RetrievalPipeline
    llm: LLMProvider
    embedder: EmbeddingProvider
    prompt: PromptTemplate


def build_app_state(config_path: str) -> AppState:
    cfg = load_run_config(config_path)
    return AppState(
        cfg=cfg,
        pipeline=RetrievalPipeline.from_config(cfg),
        llm=get_llm(cfg.generation.llm.provider, cfg.generation.llm.model),
        embedder=get_embedder(cfg.embedding.provider, cfg.embedding.model),
        prompt=PROMPTS[cfg.generation.prompt_version],
    )


def get_app_state(request: Request) -> AppState:
    """503s rather than crashing a request when startup couldn't build the
    pipeline (e.g. the baked index or an LLM API key is missing) -- the
    container itself should still come up so /api/health/ready can report
    why, instead of the whole app failing to start."""
    # getattr, not request.app.state.rag, in case lifespan never ran (e.g. a
    # TestClient used without its own context manager) -- app.state is a
    # bare `State` object that raises AttributeError on an unset attribute
    # rather than returning None.
    state: AppState | None = getattr(request.app.state, "rag", None)
    if state is None:
        raise HTTPException(status_code=503, detail="RAG backend unavailable: see /api/health/ready")
    return state
