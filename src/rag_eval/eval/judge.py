"""Artifact-anchored RAGAS judging (docs/plan.md Phase 8).

`rag-eval eval judge <run_id>` reads ``runs/<id>/generation.jsonl`` (written
by ``eval/generate.py``) and writes ``runs/<id>/judge.jsonl`` -- per-item
RAGAS scores -- plus a ``judge`` block merged into the run manifest. This is
the generalized, run-artifact-anchored form of the old
``eval/run_ragas.py`` ``--export-only`` / ``--score-only`` split, kept
working for ``notebooks/kaggle_judge_eval.py``:

    judge <run_id> --export-only     -> runs/<id>/judge_export.jsonl
                                        (portable, no scores; upload to Kaggle)
    judge <run_id>                   -> score locally with the hosted judge
    judge <run_id> --score-only F    -> merge a Kaggle-produced scored file
                                        back in, calling no judge

The judge is always hosted and, per the plan, never the generator's own
model -- ``judge_run`` asserts that and records the judge in the manifest.
``answer_relevancy.strictness`` is set to 3 (the old ``1`` was a
single-sample estimate -- hence the run of exact 0.0s in the legacy CSV).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_eval.eval.retrieval_metrics import bootstrap_ci
from rag_eval.runs.manifest import DEFAULT_RUNS_ROOT, PINNED_DIRNAME

if TYPE_CHECKING:
    from datasets import Dataset

RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
STRICTNESS = 3


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def resolve_run_dir(run_id: str, runs_root: Path = DEFAULT_RUNS_ROOT) -> Path:
    for candidate in (Path(run_id), runs_root / run_id, runs_root / PINNED_DIRNAME / run_id):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no run directory for {run_id!r} under {runs_root}")


def load_generation_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "generation.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `rag-eval eval run` with generation.enabled: true first"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_ragas_rows(generation_rows: list[dict]) -> list[dict]:
    """One RAGAS input row per generation row, using RAGAS's legacy column
    names. Rows with no ``ground_truth`` are kept -- the reference-free
    metrics (faithfulness, answer_relevancy) still score, and aggregation
    skips the NaNs the reference-based metrics produce."""
    rows = []
    for gen in generation_rows:
        rows.append(
            {
                "item_id": gen["item_id"],
                "question": gen["question"],
                "answer": gen["answer"],
                "contexts": list(gen.get("contexts", [])),
                "ground_truth": gen.get("ground_truth") or "",
            }
        )
    return rows


def dataset_to_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def jsonl_to_dataset(path: str | Path) -> Dataset:
    from datasets import Dataset

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(rows)


def score_rows(
    ragas_rows: list[dict],
    *,
    provider: str,
    model: str,
    max_workers: int = 1,
    timeout: int = 600,
) -> list[dict]:
    """Run RAGAS against a hosted judge. Returns per-item score dicts
    ``{item_id, faithfulness, answer_relevancy, context_precision,
    context_recall}`` aligned by row order."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig as RagasRunConfig

    from rag_eval.providers.langchain_adapters import build_judge

    answer_relevancy.strictness = STRICTNESS
    judge_llm, judge_embeddings = build_judge(provider=provider, model=model)
    dataset = Dataset.from_list(
        [{k: r[k] for k in ("question", "answer", "contexts", "ground_truth")} for r in ragas_rows]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RagasRunConfig(max_workers=max_workers, timeout=timeout),
    )
    frame = result.to_pandas()  # type: ignore[union-attr]  # evaluate() returns EvaluationResult here
    scored = []
    for row, (_, record) in zip(ragas_rows, frame.iterrows()):
        scored.append(
            {"item_id": row["item_id"], **{m: float(record[m]) for m in RAGAS_METRIC_NAMES}}
        )
    return scored


def load_scored_file(path: str | Path, ragas_rows: list[dict]) -> list[dict]:
    """Read a Kaggle-produced scored file (``.csv`` or ``.jsonl``) back in.
    Aligns by ``item_id`` when the file carries one, otherwise by row order
    against ``ragas_rows``."""
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    scored = []
    for i, record in enumerate(records):
        item_id = record.get("item_id") or ragas_rows[i]["item_id"]
        scored.append(
            {"item_id": item_id, **{m: float(record[m]) for m in RAGAS_METRIC_NAMES}}
        )
    return scored


def aggregate_judge(per_item: list[dict], *, bootstrap_n: int = 1000, seed: int = 0) -> dict:
    """Per-metric mean + 95% bootstrap CI, skipping NaNs (rows the judge
    failed on, or reference-based metrics on a no-reference item)."""
    out: dict[str, dict] = {}
    for metric in RAGAS_METRIC_NAMES:
        values = [r[metric] for r in per_item if metric in r and not _is_nan(r[metric])]
        if not values:
            out[metric] = {"mean": None, "ci": [None, None], "n": 0}
            continue
        lo, hi = bootstrap_ci(values, n=bootstrap_n, seed=seed)
        out[metric] = {"mean": sum(values) / len(values), "ci": [lo, hi], "n": len(values)}
    return out


def merge_judge_into_manifest(run_dir: Path, judge_block: dict) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("metrics", {})["judge"] = judge_block
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _judge_spec(manifest_config: dict, provider_override: str | None, model_override: str | None):
    judge_cfg = manifest_config.get("eval", {}).get("judge", {})
    provider = provider_override or judge_cfg.get("provider", "gemini")
    model = model_override or judge_cfg.get("model", "gemini-2.5-flash")
    max_workers = int(judge_cfg.get("max_workers", 1))

    gen_llm = manifest_config.get("generation", {}).get("llm", {})
    if (provider, model) == (gen_llm.get("provider"), gen_llm.get("model")):
        raise ValueError(
            f"judge ({provider}/{model}) must not be the generator's own model "
            "(docs/plan.md Phase 8) -- pass --judge-provider/--judge-model or fix eval.judge"
        )
    return provider, model, max_workers


def judge_run(
    run_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    export_only: bool = False,
    score_only: str | Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> Path:
    run_dir = resolve_run_dir(run_id, runs_root)
    manifest_config = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get(
        "config", {}
    )
    ragas_rows = to_ragas_rows(load_generation_rows(run_dir))

    if export_only:
        export_path = run_dir / "judge_export.jsonl"
        dataset_to_jsonl(ragas_rows, export_path)
        return export_path

    provider, model, max_workers = _judge_spec(manifest_config, provider_override, model_override)

    if score_only is not None:
        per_item = load_scored_file(score_only, ragas_rows)
    else:
        per_item = score_rows(
            ragas_rows, provider=provider, model=model, max_workers=max_workers
        )

    judge_path = run_dir / "judge.jsonl"
    dataset_to_jsonl(per_item, judge_path)

    merge_judge_into_manifest(
        run_dir,
        {
            "provider": provider,
            "model": model,
            "strictness": STRICTNESS,
            "n_items": len(per_item),
            "scored_offline": score_only is not None,
            "aggregates": aggregate_judge(per_item),
        },
    )
    return judge_path
