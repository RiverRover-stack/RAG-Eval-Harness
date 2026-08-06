"""Groq chat completion via plain httpx (OpenAI-compatible endpoint)."""

from __future__ import annotations

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import LLMResponse

_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = settings.groq_api_key if api_key is None else api_key
        if not self._api_key:
            raise ValueError("Groq provider requires GROQ_API_KEY to be set in .env")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = httpx.post(
            _API_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
