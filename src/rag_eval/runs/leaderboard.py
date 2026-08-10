"""Turn a pile of run manifests into a leaderboard, and compare two runs
without accidentally attributing a generator change to retrieval (or vice
versa) -- see the frozen-generator protocol in docs/plan.md C2.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from rag_eval.runs.manifest import RunManifest

CONFOUNDED: Final = "CONFOUNDED"

DEFAULT_METRICS = ("recall_at_5", "mrr", "ndcg_at_10")


def build_leaderboard(
    runs: Sequence[RunManifest],
    *,
    dataset: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[dict]:
    """One row per run, each metric alongside its bootstrap CI, newest
    first. `dataset` picks which of a run's (possibly several) scored
    datasets to report -- a run's docs_synth_v1 and discussions_v2 numbers
    aren't comparable to each other, so mixing them in one table would be
    the leaderboard lying by omission."""
    rows = []
    for run in runs:
        block = run.metrics.get(dataset)
        row = {
            "run_id": run.run_id,
            "name": run.name,
            "config_hash": run.config_hash,
            "retrieval_hash": run.retrieval_hash,
            "generation_hash": run.generation_hash,
            "created_at": run.created_at,
            "git_sha": run.git_sha,
            "n": block.get("n") if block else 0,
        }
        for metric in metrics:
            value = run.get_metric(dataset, metric)
            ci = block.get("cis", {}).get(metric) if block else None
            row[metric] = value
            row[f"{metric}_ci"] = ci
        rows.append(row)
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows


def write_leaderboard(
    runs: Sequence[RunManifest],
    out_path: Path,
    *,
    dataset: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Path:
    rows = build_leaderboard(runs, dataset=dataset, metrics=metrics)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def compare_runs(
    base: RunManifest,
    cand: RunManifest,
    *,
    dataset: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict | Literal["CONFOUNDED"]:
    """A delta is only meaningful if the generator didn't change underneath
    it. If it did -- even for a retrieval-only run where generation is
    disabled but its config drifted -- this returns CONFOUNDED instead of a
    number that looks like a retrieval result but isn't one."""
    if base.generation_hash != cand.generation_hash:
        return CONFOUNDED

    deltas = {}
    for metric in metrics:
        base_value = base.get_metric(dataset, metric)
        cand_value = cand.get_metric(dataset, metric)
        delta = (
            cand_value - base_value if base_value is not None and cand_value is not None else None
        )
        deltas[metric] = {"base": base_value, "cand": cand_value, "delta": delta}
    return deltas


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gold_rank(row: dict) -> int | None:
    gold = set(row.get("gold_chunk_ids", []))
    for candidate in row.get("retrieved", []):
        if candidate["chunk_id"] in gold:
            return candidate["rank"]
    return None


def regressed_items(
    base_run_dir: Path,
    cand_run_dir: Path,
    *,
    metric: str,
) -> list[dict]:
    """Per-item regressions between two runs' retrieval.jsonl, worst first.
    A leaderboard delta says "recall@5 dropped 3 points"; this says exactly
    which questions caused it and how far their gold chunk fell in rank --
    the part of an eval dashboard that's actually actionable."""
    base_rows = {row["item_id"]: row for row in _read_jsonl(base_run_dir / "retrieval.jsonl")}
    cand_rows = {row["item_id"]: row for row in _read_jsonl(cand_run_dir / "retrieval.jsonl")}

    regressions = []
    for item_id, base_row in base_rows.items():
        cand_row = cand_rows.get(item_id)
        if cand_row is None:
            continue
        base_value = base_row.get("metrics", {}).get(metric)
        cand_value = cand_row.get("metrics", {}).get(metric)
        if base_value is None or cand_value is None or cand_value >= base_value:
            continue
        regressions.append(
            {
                "item_id": item_id,
                "question": base_row.get("question", ""),
                "metric": metric,
                "base_value": base_value,
                "cand_value": cand_value,
                "delta": cand_value - base_value,
                "base_gold_rank": _gold_rank(base_row),
                "cand_gold_rank": _gold_rank(cand_row),
            }
        )
    regressions.sort(key=lambda r: r["delta"])
    return regressions
