"""Answer generation via a local Ollama LLM, grounded in retrieved chunks."""

import ollama

from rag_eval.common.config import settings
from rag_eval.common.schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer FastAPI questions using only the provided context, which is "
    "drawn from GitHub Discussions. If the context does not contain the answer, "
    "say you don't know rather than guessing."
)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n---\n\n".join(c.content for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    client = ollama.Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.ollama_llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    return response["message"]["content"]
