"""Gemini chat completion via plain httpx (generateContent REST endpoint)."""

from __future__ import annotations

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import LLMResponse

_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _to_gemini_contents(messages: list[dict[str, str]]) -> tuple[str | None, list[dict]]:
    """Split OpenAI-style messages into (system_instruction, contents).

    Gemini has no "system" role in `contents`; a single system message (the
    only kind this project's prompts use) becomes `system_instruction`.
    """
    system: str | None = None
    contents = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
            continue
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return system, contents


class GeminiLLM:
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = settings.gemini_api_key if api_key is None else api_key
        if not self._api_key:
            raise ValueError("Gemini provider requires GEMINI_API_KEY to be set in .env")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        system, contents = _to_gemini_contents(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system is not None:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        response = httpx.post(
            _API_URL.format(model=self.model),
            params={"key": self._api_key},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=data["candidates"][0]["content"]["parts"][0]["text"],
            model=self.model,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )
