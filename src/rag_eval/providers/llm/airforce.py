"""Groq chat completion via plain httpx (OpenAI-compatible endpoint)."""

from __future__ import annotations

import time

import httpx

from rag_eval.common.config import settings
from rag_eval.providers.base import LLMResponse

_API_URL = "https://api.airforce/v1/chat/completions"

# The free tier's per-minute limit is easy to hit from a batch job (e.g.
# generating ~150 synthetic questions back to back -- docs/plan.md Phase 4)
# well before any daily quota. A 429 there is transient, not a real
# failure, so it's worth a short, bounded retry instead of surfacing as a
# generation error and silently shrinking the eval set.
_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 2.0

# This provider's account is rate-limited to 1 request/minute. Padding to
# 90s (rather than exactly 60s) leaves headroom for request latency and
# clock drift so a run doesn't tip over the limit and start eating 429s
# it can't fully out-wait within _MAX_RETRIES.
_MIN_REQUEST_INTERVAL_SECONDS = 90.0


class AirForceLLM:
    name = "airforce"

    def __init__(self, model: str = "ministral-14b-latest", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = settings.airforce_api_key if api_key is None else api_key
        if not self._api_key:
            raise ValueError("Airforce provider requires AIRFORCE_API_KEY to be set in .env")
        self._last_request_at: float | None = None

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

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _post_with_retry(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
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
