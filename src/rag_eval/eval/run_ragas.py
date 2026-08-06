"""
Run the RAG pipeline over the eval set and score it with RAGAS.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall.

RAGAS needs an LLM/embeddings for judging. Two judges are supported via
RAGAS_JUDGE in .env:

  - "ollama" (default): local ChatOllama, matches the pipeline's own model.
    Slow on CPU-only hardware -- each metric call queues behind a single
    model instance, so max_workers is kept at 1 and the timeout generous
    (see ragas_max_workers / ragas_timeout in config.py).
  - "groq": hosted Llama via Groq's free tier. Useful for validating the
    harness/dataset quickly without being bottlenecked by local compute.
    Requires GROQ_API_KEY in .env. Embeddings stay on local Ollama either
    way -- nomic-embed-text is cheap and isn't what caused the timeouts.

Usage:
    uv run python -m rag_eval.eval.run_ragas
"""

import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from rag_eval.common.config import settings
from rag_eval.providers.langchain_adapters import build_judge
from rag_eval.rag.pipeline import answer_question


def load_eval_set() -> list[dict]:
    with open(settings.eval_set_path, encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    if settings.eval_sample_limit:
        examples = examples[: settings.eval_sample_limit]
    return examples


def run_pipeline_over_eval_set(examples: list[dict]) -> Dataset:
    rows = []
    for ex in examples:
        result = answer_question(ex["question"])
        rows.append(
            {
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
            }
        )
    return Dataset.from_list(rows)


def score(dataset: Dataset):
    judge_llm, judge_embeddings = build_judge()
    run_config = RunConfig(
        max_workers=settings.ragas_max_workers,
        timeout=settings.ragas_timeout,
    )
    answer_relevancy.strictness = 1
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )


if __name__ == "__main__":
    examples = load_eval_set()
    dataset = run_pipeline_over_eval_set(examples)
    results = score(dataset)
    print(results)
    results.to_pandas().to_csv("./data/eval_sets/ragas_results.csv", index=False)
