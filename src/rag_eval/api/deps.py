"""FastAPI dependency wiring for the Phase 7 app factory (docs/plan.md).

`build_app_state` does the heavy lifting once, at lifespan startup --
loading the `RunConfig` and building the `RetrievalPipeline` from it (which
in turn warms the embedder, reranker, and BM25 index), plus resolving the
serving LLM and prompt template -- so no request ever pays for any of that.
`get_app_state` is the per-request dependency every route pulls it back
through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException, Request

from rag_eval.config.run_config import RunConfig, load_run_config
from rag_eval.eval.datasets import load_dataset
from rag_eval.eval.gold import EvalItem, GoldIndex, resolve_gold_chunks
from rag_eval.eval.runner import build_corpus_gold_index
from rag_eval.providers import get_embedder, get_llm
from rag_eval.providers.base import EmbeddingProvider, LLMProvider
from rag_eval.rag.prompts import PROMPTS
from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.retrieval.pipeline import RetrievalPipeline

# Eval sets whose questions back the "Try" suggestion chips and gold-aware
# citation highlighting -- docs_synth_v1 first so suggestions() can prefer it.
EVAL_DATASETS = ("docs_synth_v1", "discussions_v2")


def _empty_gold_index() -> GoldIndex:
    return GoldIndex(by_url={}, by_page={})


@dataclass
class AppState:
    cfg: RunConfig
    pipeline: RetrievalPipeline
    llm: LLMProvider
    embedder: EmbeddingProvider
    prompt: PromptTemplate
    # keyed by exact question.strip() -- lets /api/ask* mark a citation
    # `gold` and /api/suggestions offer only gold-resolvable questions,
    # without fabricating either for a free-typed question. Default empty so
    # existing AppState(...) call sites (tests) don't have to know about it.
    gold_index: GoldIndex = field(default_factory=_empty_gold_index)
    eval_items_by_question: dict[str, EvalItem] = field(default_factory=dict)


def _load_eval_items_by_question(gold_index: GoldIndex) -> dict[str, EvalItem]:
    items_by_question: dict[str, EvalItem] = {}
    for dataset_name in EVAL_DATASETS:
        for item in load_dataset(dataset_name):
            resolved = resolve_gold_chunks(item, gold_index)
            items_by_question[resolved.question.strip()] = resolved
    return items_by_question


def build_app_state(config_path: str) -> AppState:
    cfg = load_run_config(config_path)
    gold_index = build_corpus_gold_index()
    return AppState(
        cfg=cfg,
        pipeline=RetrievalPipeline.from_config(cfg),
        llm=get_llm(cfg.generation.llm.provider, cfg.generation.llm.model),
        embedder=get_embedder(cfg.embedding.provider, cfg.embedding.model),
        prompt=PROMPTS[cfg.generation.prompt_version],
        gold_index=gold_index,
        eval_items_by_question=_load_eval_items_by_question(gold_index),
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
