"""Groq chat completion via plain httpx (OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import LLMResponse, StreamChunk

_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# The free tier's per-minute limit is easy to hit from a batch job (e.g.
# generating ~150 synthetic questions back to back -- docs/plan.md Phase 4)
# well before any daily quota. A 429 there is transient, not a real
# failure, so it's worth a short, bounded retry instead of surfacing as a
# generation error and silently shrinking the eval set.
_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 2.0


class GroqLLM:
    name = "groq"

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None) -> None:
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
        response = self._post_with_retry(messages, temperature=temperature, max_tokens=max_tokens)
        data = response.json()
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def astream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """Groq's stream is OpenAI-compatible SSE: lines prefixed `data: `,
        each a JSON chunk with `choices[0].delta.content`, terminated by a
        literal `data: [DONE]`. `stream_options.include_usage` asks for a
        final usage-only chunk (no `[DONE]` retry loop here -- a stream that
        429s mid-flight would need to restart from scratch anyway, so the
        batch-job retry logic above doesn't apply)."""
        async with httpx.AsyncClient(timeout=60) as client, client.stream(
            "POST",
            _API_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            response.raise_for_status()
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                usage = data.get("usage") or data.get("x_groq", {}).get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                choices = data.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    yield StreamChunk(delta=delta)
            yield StreamChunk(
                delta="", done=True, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )

    def _post_with_retry(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
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
            if response.status_code != 429 or attempt == _MAX_RETRIES:
                response.raise_for_status()
                return response
            wait = self._retry_after(response) or _BASE_BACKOFF_SECONDS * (2**attempt)
            time.sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        header = response.headers.get("retry-after")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None
