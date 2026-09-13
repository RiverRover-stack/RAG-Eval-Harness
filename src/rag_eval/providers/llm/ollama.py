"""Local Ollama chat completion via plain httpx (no `ollama` package)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import LLMResponse, StreamChunk


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
            timeout=settings.ollama_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def astream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """Ollama's stream is NDJSON, not SSE -- one bare JSON object per
        line, no `data: ` prefix, no `[DONE]` sentinel. The last line has
        `"done": true` plus the final `prompt_eval_count`/`eval_count`."""
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client, client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                delta = data.get("message", {}).get("content") or ""
                if delta:
                    yield StreamChunk(delta=delta)
                if data.get("done"):
                    yield StreamChunk(
                        delta="",
                        done=True,
                        prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"),
                    )
