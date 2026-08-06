"""Local Ollama chat completion via plain httpx (no `ollama` package)."""

from __future__ import annotations

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import LLMResponse


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str = "fdm-llama", base_url: str | None = None) -> None:
        self.model = model
        self._base_url = base_url or settings.ollama_base_url

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=settings.ragas_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
