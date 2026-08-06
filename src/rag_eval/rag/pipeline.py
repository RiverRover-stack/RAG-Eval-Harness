"""End-to-end RAG pipeline: question -> retrieve -> generate -> answer."""

from rag_eval.providers.base import LLMProvider
from rag_eval.rag.generator import generate_answer
from rag_eval.rag.retriever import retrieve


def answer_question(question: str, k: int = 5, llm: LLMProvider | None = None) -> dict:
    chunks = retrieve(question, k=k)
    answer = generate_answer(question, chunks, llm=llm)
    return {
        "question": question,
        "answer": answer,
        "contexts": [c.content for c in chunks],
        "sources": [c.source_id for c in chunks],
    }
