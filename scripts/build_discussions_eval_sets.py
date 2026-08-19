"""Derive discussions_v2 and discussions_gen_v1 from the raw 27-row GitHub
Discussions eval set (docs/plan.md Phase 4).

Filters out the "unanswerable" rows -- ground truths under 200 chars are
almost always social-closure text ("Thanks, tracking it in #123") rather
than an actual answer, and dragged every aggregate down when scored as-is
(CLAUDE.md / plan problem set). What's left still has no gold docs URLs;
those get filled in by a human via `rag-eval eval label --dataset
discussions_v2` -- this script only does the mechanical filtering and
exclude_chunk_ids bookkeeping, not the judgment call of "which docs section
answers this."

discussions_v2 and discussions_gen_v1 start out identical (same question,
same ground truth, same self-retrieval exclusion) and only differ in the
`dataset` tag -- v2 is the retrieval-eval bridge (real questions, hand-
labeled retrieval targets), gen_v1 is for answer-quality scoring only and
never needs gold_urls at all.

Usage:
    uv run python scripts/build_discussions_eval_sets.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval.eval.gold import EvalItem, build_gold_index

RAW_PATH = Path("data/eval_sets/fastapi_discussions_eval.jsonl")
OUT_V2 = Path("data/eval_sets/discussions_v2.jsonl")
OUT_GEN_V1 = Path("data/eval_sets/discussions_gen_v1.jsonl")

MIN_ANSWERABLE_CHARS = 200


def _discussion_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _write_jsonl(items: list[EvalItem], path: Path) -> None:
    path.write_text("\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8")


def build() -> None:
    from rag_eval.ingestion.chunker import qa_to_chunks
    from rag_eval.ingestion.discussions_snapshot import load_snapshot

    raw_rows = [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    discussion_chunks = [c for qa in load_snapshot() for c in qa_to_chunks(qa)]
    gold_index = build_gold_index(discussion_chunks)

    kept = [row for row in raw_rows if len(row["ground_truth"].strip()) >= MIN_ANSWERABLE_CHARS]
    dropped = len(raw_rows) - len(kept)
    print(f"{len(raw_rows)} raw rows, {dropped} dropped as unanswerable (<{MIN_ANSWERABLE_CHARS} chars), {len(kept)} kept")

    v2_items: list[EvalItem] = []
    gen_items: list[EvalItem] = []
    for row in kept:
        item_id = _discussion_id_from_url(row["source_url"])
        own_chunk_ids = sorted(gold_index.by_url.get(row["source_url"], set()))

        v2_items.append(
            EvalItem(
                id=item_id,
                dataset="discussions_v2",
                question=row["question"],
                ground_truth=row["ground_truth"],
                gold_urls=[],
                exclude_chunk_ids=own_chunk_ids,
                provenance=row["source_url"],
            )
        )
        gen_items.append(
            EvalItem(
                id=item_id,
                dataset="discussions_gen_v1",
                question=row["question"],
                ground_truth=row["ground_truth"],
                gold_urls=[],
                exclude_chunk_ids=own_chunk_ids,
                provenance=row["source_url"],
            )
        )

    _write_jsonl(v2_items, OUT_V2)
    _write_jsonl(gen_items, OUT_GEN_V1)
    print(f"wrote {len(v2_items)} items to {OUT_V2}")
    print(f"wrote {len(gen_items)} items to {OUT_GEN_V1}")
    print("gold_urls are empty -- run: uv run rag-eval eval label --dataset discussions_v2")


if __name__ == "__main__":
    build()
