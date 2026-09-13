"""LangChain object construction, confined to the RAGAS judge.

The serving path (rag/generator.py) and everything else in providers/ use
plain httpx -- light, streamable, no LangChain churn in the hot path. RAGAS
itself requires LangChain LLM/embeddings objects to score with, so this
module is the one deliberate exception. See
docs/adr/0004-plain-httpx-providers-langchain-confined-to-judge.md.

Moved here from eval/run_ragas.py verbatim (Phase 2), then generalized to
take an explicit provider/model in Phase 8 once eval/judge.py made
RunConfig.eval.judge the only source of "which judge" -- the env-based
RAGAS_JUDGE selector this used to fall back to is gone.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import SecretStr

from rag_eval.common.config import settings


def build_judge(provider: str, model: str):
    """Return (llm, embeddings) for the judge.

    `provider` / `model` come from `RunConfig.eval.judge` (see
    eval/judge.py, eval/rubric.py) -- the single place "which judge" is
    decided. API keys always come from settings. Judge embeddings stay on
    local Ollama regardless -- cheap, and not what caused the timeouts.
    """
    judge_embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "judge provider 'groq' requires GROQ_API_KEY to be set in .env "
                "(get a free key at https://console.groq.com)"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=SecretStr(settings.groq_api_key),
            temperature=0,
            max_tokens=4096,
        ), judge_embeddings

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "judge provider 'gemini' requires GEMINI_API_KEY to be set in .env "
                "(get a free key at https://aistudio.google.com/apikey)"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=SecretStr(settings.gemini_api_key),
            temperature=0,
            max_tokens=4096,
            # Native JSON mode: forces the model to emit only valid JSON,
            # which is what caused OutputParserException on the small Groq
            # model (it would wrap JSON in explanatory prose).
            generation_config={"response_mime_type": "application/json"},
        ), judge_embeddings

    if provider != "ollama":
        raise ValueError(
            f"Unknown judge provider {provider!r}, expected 'ollama', 'groq', or 'gemini'"
        )

    return ChatOllama(model=model, base_url=settings.ollama_base_url), judge_embeddings
