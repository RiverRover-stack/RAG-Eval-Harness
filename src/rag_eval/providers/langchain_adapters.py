"""LangChain object construction, confined to the RAGAS judge.

The serving path (rag/generator.py) and everything else in providers/ use
plain httpx -- light, streamable, no LangChain churn in the hot path. RAGAS
itself requires LangChain LLM/embeddings objects to score with, so this
module is the one deliberate exception. See
docs/adr/0004-plain-httpx-providers-langchain-confined-to-judge.md.

Moved here from eval/run_ragas.py verbatim (Phase 2); behavior unchanged.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_eval.common.config import settings


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
