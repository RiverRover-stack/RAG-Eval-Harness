"""
Build a RAGAS-ready eval set (JSONL) from the pinned GitHub Discussions
snapshot (see ingestion/discussions_snapshot.py) -- not a live fetch, so the
eval set and whatever index it's scored against come from the same frozen
snapshot instead of two fetches taken on different days
(docs/plan.md problem 3).

Each row: {question, ground_truth, source_url}. Contexts/answer get filled
in later at eval-run time by actually running the RAG pipeline (see
eval/run_ragas.py), so the eval set here stays pipeline-agnostic.

Usage:
    uv run python -m rag_eval.eval.build_eval_set
"""

from pathlib import Path

from rag_eval.common.config import settings
from rag_eval.common.schemas import DiscussionQA, EvalExample
from rag_eval.ingestion.discussions_snapshot import DEFAULT_SNAPSHOT_PATH, load_snapshot


def qa_to_eval_example(qa: DiscussionQA) -> EvalExample:
    """The query is the title plus the question body's head, not the bare
    title -- discussion titles are often terse bug-report headlines (median
    79 chars across the current eval set) that discard the actual detail
    written in the body."""
    body_head = qa.question_body[:500]
    question = f"{qa.title}\n\n{body_head}" if body_head else qa.title
    return EvalExample(
        question=question,
        ground_truth=qa.answer_body,
        source_url=qa.url,
    )


def build_and_save(snapshot_path: Path = DEFAULT_SNAPSHOT_PATH) -> str:
    qas = load_snapshot(snapshot_path)
    examples = [qa_to_eval_example(qa) for qa in qas]

    with open(settings.eval_set_path, "w", encoding="utf-8") as f:
        f.writelines(ex.model_dump_json() + "\n" for ex in examples)

    return settings.eval_set_path


if __name__ == "__main__":
    path = build_and_save()
    print(f"Wrote eval set to {path}")
