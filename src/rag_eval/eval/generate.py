"""Batch answer generation for an eval run (docs/plan.md Phase 8).

The judge (`eval/judge.py`, `eval/rubric.py`) scores a run by reading
`runs/<id>/generation.jsonl` -- so something has to write it. This module is
that step: for every eval item, retrieve context, generate a v2-cited
answer, and score it for citation validity + embedding groundedness (the
same logic `api/routes/ask.py` uses, via `rag.generator.generate_cited_answer`).

It runs only when `cfg.generation.enabled` and is kept strictly separate
from `eval/runner.py`'s retrieval scoring: retrieval metrics never invoke
the generator (docs/plan.md C2, mechanism 1). The default generate fn does
its own retrieval so the row is self-contained -- the pipeline is
deterministic at `temperature: 0`, so the context it sees matches what the
retrieval pass scored.

`GenerateFn` is injectable so tests exercise the runner wiring with a
scripted fake and never touch a live LLM or Chroma.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict

from rag_eval.config.run_config import RunConfig
from rag_eval.eval.gold import EvalItem

GenerateFn = Callable[[EvalItem, RunConfig], dict]


def _default_generate_fn(cfg: RunConfig) -> GenerateFn:
    from rag_eval.providers import get_embedder, get_llm
    from rag_eval.rag.generator import generate_cited_answer
    from rag_eval.rag.prompts import PROMPTS
    from rag_eval.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline.from_config(cfg)
    llm = get_llm(cfg.generation.llm.provider, cfg.generation.llm.model)
    embedder = get_embedder(cfg.embedding.provider, cfg.embedding.model)
    try:
        prompt = PROMPTS[cfg.generation.prompt_version]
    except KeyError:
        raise ValueError(
            f"unknown generation.prompt_version {cfg.generation.prompt_version!r}; "
            f"known: {sorted(PROMPTS)}"
        ) from None

    def generate(item: EvalItem, run_cfg: RunConfig) -> dict:
        result = pipeline.retrieve(item.question, k=run_cfg.retrieval.top_k)
        started = time.perf_counter()
        gen = generate_cited_answer(
            item.question,
            result.candidates,
            prompt=prompt,
            llm=llm,
            embedder=embedder,
            abstain_below=run_cfg.generation.groundedness.abstain_below,
            temperature=run_cfg.generation.llm.temperature,
            max_tokens=run_cfg.generation.llm.max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return build_row(item, run_cfg, gen, result.candidates, latency_ms)

    return generate


def build_row(item, run_cfg: RunConfig, gen, candidates, latency_ms: float) -> dict:
    """Assemble one `generation.jsonl` row. `contexts` + `ground_truth` are
    stored here so the judge never has to re-retrieve or re-generate."""
    return {
        "item_id": f"{item.dataset}::{item.id}",
        "dataset": item.dataset,
        "question": item.question,
        "ground_truth": item.ground_truth,
        "prompt_version": run_cfg.generation.prompt_version,
        "answer": gen.answer,
        "abstained": gen.abstained,
        "groundedness": gen.groundedness,
        "citations": {
            "citations": [asdict(c) for c in gen.citations.citations],
            "unknown_indices": list(gen.citations.unknown_indices),
            "uncited_sentences": list(gen.citations.uncited_sentences),
            "coverage": gen.citations.coverage,
        },
        "usage": {
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
        },
        "latency_ms": latency_ms,
        "contexts": [c.content for c in candidates],
        "context_urls": [c.url for c in candidates],
        "context_chunk_ids": [c.chunk_id for c in candidates],
    }


def generate_over_datasets(
    cfg: RunConfig,
    items_by_dataset: dict[str, list[EvalItem]],
    generate_fn: GenerateFn | None = None,
) -> list[dict]:
    generate = generate_fn or _default_generate_fn(cfg)
    rows: list[dict] = []
    for items in items_by_dataset.values():
        for item in items:
            rows.append(generate(item, cfg))
    return rows
