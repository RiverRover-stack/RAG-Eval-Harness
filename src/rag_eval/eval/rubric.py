"""Structured rubric judging with a deterministic cross-check (docs/plan.md
Phase 8).

`rag-eval eval rubric <run_id>` makes one structured judge call per item in
``runs/<id>/generation.jsonl`` scoring five dimensions 1-5 with a required
justification, and writes ``runs/<id>/rubric.jsonl`` plus a ``rubric`` block
in the manifest.

Alongside the judge's ``citation_accuracy`` (1-5), a *deterministic*
citation accuracy is computed from the embedding-support scorer: for each
``[n]`` marker, does the sentence it sits in actually match the chunk it
cites? The judge-vs-deterministic correlation is reported -- a low
correlation is itself a finding worth writing up (plan Phase 8, Verify).

All five dimensions are scored so that **higher is better** (hallucination
5 = fully grounded, 1 = major fabrication), so the means read consistently.
The judge is hosted and, per the plan, never the generator's own model.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_eval.eval.judge import (
    _judge_spec,
    load_generation_rows,
    merge_judge_into_manifest,
    resolve_run_dir,
)
from rag_eval.rag.citations import split_sentences
from rag_eval.runs.manifest import DEFAULT_RUNS_ROOT

RUBRIC_DIMENSIONS = ("correctness", "completeness", "citation_accuracy", "hallucination", "tone")
CITATION_SUPPORT_THRESHOLD = 0.45

SYSTEM_PROMPT = (
    "You are a strict grader of FastAPI question-answering. Grade only what is "
    "shown. Respond with a single JSON object and nothing else."
)

_RUBRIC_INSTRUCTIONS = """Score each dimension from 1 (worst) to 5 (best):
- correctness: is the answer factually right for the question?
- completeness: does it cover what the question actually asked?
- citation_accuracy: do the [n] markers point at context that supports the sentence they follow? (5 if there are no unsupported citations; if the answer has no citations at all, score 3)
- hallucination: 5 = every claim is grounded in the context; 1 = major invented facts.
- tone: is it clear, direct, and appropriately concise?

Return exactly:
{"correctness": int, "completeness": int, "citation_accuracy": int, "hallucination": int, "tone": int, "justification": "one or two sentences"}"""


@dataclass
class RubricScore:
    item_id: str
    scores: dict[str, int]
    justification: str
    deterministic_citation_accuracy: float | None
    n_citations: int


def build_rubric_messages(gen_row: dict) -> list[dict]:
    blocks = "\n".join(
        f"[{i}] {c}" for i, c in enumerate(gen_row.get("contexts", []), start=1)
    )
    user = (
        f"Question:\n{gen_row['question']}\n\n"
        f"Reference answer (may be empty):\n{gen_row.get('ground_truth') or '(none)'}\n\n"
        f"Numbered context the answer was generated from:\n{blocks}\n\n"
        f"Answer under review:\n{gen_row['answer']}\n\n"
        f"{_RUBRIC_INSTRUCTIONS}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_rubric_response(text: str) -> tuple[dict[str, int], str]:
    """Extract the JSON object even if the model wrapped it in prose or
    ```json fences. Raises ValueError on anything unparseable or missing a
    dimension."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in rubric response: {text[:200]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed rubric JSON: {exc}") from exc

    scores: dict[str, int] = {}
    for dim in RUBRIC_DIMENSIONS:
        if dim not in payload:
            raise ValueError(f"rubric response missing {dim!r}")
        try:
            value = int(payload[dim])
        except (TypeError, ValueError):
            raise ValueError(f"rubric {dim!r} not an int: {payload[dim]!r}") from None
        if not 1 <= value <= 5:
            raise ValueError(f"rubric {dim!r} out of range 1-5: {value}")
        scores[dim] = value
    return scores, str(payload.get("justification", ""))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def deterministic_citation_accuracy(
    gen_row: dict, embedder, threshold: float = CITATION_SUPPORT_THRESHOLD
) -> tuple[float | None, int]:
    """Fraction of `[n]` markers whose enclosing sentence matches the chunk
    they cite (cosine >= threshold). None when the answer has no citations."""
    citations = gen_row.get("citations", {}).get("citations", [])
    contexts = gen_row.get("contexts", [])
    if not citations:
        return None, 0

    sentences = split_sentences(gen_row["answer"])
    supported = 0
    for cite in citations:
        idx = cite["index"]
        if not 1 <= idx <= len(contexts):
            continue
        sentence = next(
            (s for s, start, end in sentences if start <= cite["char_start"] <= end),
            None,
        )
        if sentence is None:
            continue
        sim = _cosine(embedder.embed_query(sentence), embedder.embed_query(contexts[idx - 1]))
        if sim >= threshold:
            supported += 1
    return supported / len(citations), len(citations)


def score_item(gen_row: dict, *, llm, embedder, threshold: float = CITATION_SUPPORT_THRESHOLD) -> RubricScore:
    response = llm.complete(build_rubric_messages(gen_row), temperature=0.0)
    scores, justification = parse_rubric_response(response.content)
    det, n_cites = deterministic_citation_accuracy(gen_row, embedder, threshold)
    return RubricScore(
        item_id=gen_row["item_id"],
        scores=scores,
        justification=justification,
        deterministic_citation_accuracy=det,
        n_citations=n_cites,
    )


def judge_vs_deterministic_correlation(rubric_scores: list[RubricScore]) -> float | None:
    """Pearson r between the judge's citation_accuracy (1-5) and the
    deterministic fraction (0-1), over items that have citations. None if
    fewer than two such items or no variance."""
    pairs = [
        (s.scores["citation_accuracy"], s.deterministic_citation_accuracy)
        for s in rubric_scores
        if s.deterministic_citation_accuracy is not None
    ]
    if len(pairs) < 2:
        return None
    judge = [p[0] for p in pairs]
    det = [p[1] for p in pairs]
    try:
        return statistics.correlation(judge, det)
    except statistics.StatisticsError:
        return None


def _dimension_means(rubric_scores: list[RubricScore]) -> dict[str, float]:
    return {
        dim: statistics.fmean(s.scores[dim] for s in rubric_scores) if rubric_scores else 0.0
        for dim in RUBRIC_DIMENSIONS
    }


def rubric_run(
    run_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    provider_override: str | None = None,
    model_override: str | None = None,
    threshold: float = CITATION_SUPPORT_THRESHOLD,
    llm=None,
    embedder=None,
) -> Path:
    run_dir = resolve_run_dir(run_id, runs_root)
    manifest_config = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get(
        "config", {}
    )
    provider, model, _max_workers = _judge_spec(manifest_config, provider_override, model_override)

    if llm is None:
        from rag_eval.providers import get_llm

        llm = get_llm(provider, model)
    if embedder is None:
        from rag_eval.providers import get_embedder

        emb_cfg = manifest_config.get("embedding", {})
        embedder = get_embedder(
            emb_cfg.get("provider", "fastembed"), emb_cfg.get("model", "BAAI/bge-small-en-v1.5")
        )

    rubric_scores = [
        score_item(row, llm=llm, embedder=embedder, threshold=threshold)
        for row in load_generation_rows(run_dir)
    ]

    rubric_path = run_dir / "rubric.jsonl"
    with open(rubric_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(asdict(s)) + "\n" for s in rubric_scores)

    merge_judge_into_manifest(
        run_dir,
        {
            "provider": provider,
            "model": model,
            "n_items": len(rubric_scores),
            "dimension_means": _dimension_means(rubric_scores),
            "judge_vs_deterministic_citation_accuracy_r": judge_vs_deterministic_correlation(
                rubric_scores
            ),
        },
        key="rubric",
    )
    return rubric_path
