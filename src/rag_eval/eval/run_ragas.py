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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from rag_eval.common.config import settings
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


def build_judge():
    """Return (llm, embeddings) for the judge configured via RAGAS_JUDGE."""
    judge_embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )

    if settings.ragas_judge == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "RAGAS_JUDGE=groq requires GROQ_API_KEY to be set in .env "
                "(get a free key at https://console.groq.com)"
            )
        from langchain_groq import ChatGroq

        judge_llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=4096,
        )
        return judge_llm, judge_embeddings

    if settings.ragas_judge == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "RAGAS_JUDGE=gemini requires GEMINI_API_KEY to be set in .env "
                "(get a free key at https://aistudio.google.com/apikey)"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        judge_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_tokens=4096,
            # Native JSON mode: forces the model to emit only valid JSON,
            # which is what caused OutputParserException on the small Groq
            # model (it would wrap JSON in explanatory prose).
            generation_config={"response_mime_type": "application/json"},
        )
        return judge_llm, judge_embeddings

    if settings.ragas_judge != "ollama":
        raise ValueError(
            f"Unknown RAGAS_JUDGE={settings.ragas_judge!r}, "
            "expected 'ollama', 'groq', or 'gemini'"
        )

    judge_llm = ChatOllama(model=settings.ollama_llm_model, base_url=settings.ollama_base_url)
    return judge_llm, judge_embeddings


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
