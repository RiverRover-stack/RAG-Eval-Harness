"""
Build a RAGAS-ready eval set (JSONL) from GitHub Discussions Q&A pairs.

Each row: {question, ground_truth, source_url}. Contexts/answer get filled
in later at eval-run time by actually running the RAG pipeline (see
eval/run_ragas.py), so the eval set here stays pipeline-agnostic.

Usage:
    uv run python -m rag_eval.eval.build_eval_set
"""

import json

from rag_eval.common.config import settings
from rag_eval.common.schemas import DiscussionQA, EvalExample
from rag_eval.ingestion.github_discussions import fetch_discussion_qas


def qa_to_eval_example(qa: DiscussionQA) -> EvalExample:
    return EvalExample(
        question=qa.title,
        ground_truth=qa.answer_body,
        source_url=qa.url,
    )


def build_and_save(max_pages: int | None = None) -> str:
    qas = fetch_discussion_qas(max_pages=max_pages)
    examples = [qa_to_eval_example(qa) for qa in qas]

    with open(settings.eval_set_path, "w", encoding='utf-8') as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    return settings.eval_set_path


if __name__ == "__main__":
    path = build_and_save(max_pages=settings.ingest_discussion_pages or None)
    print(f"Wrote eval set to {path}")
