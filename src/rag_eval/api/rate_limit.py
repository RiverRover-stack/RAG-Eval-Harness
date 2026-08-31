"""Per-IP token-bucket rate limiting and a global daily cost cap for
/api/ask* (docs/plan.md Phase 7): a public demo running on a real API key
gets scraped, and these are the guardrails -- not a full API-gateway
rewrite, ~50 lines that read as production thinking.

`TokenBucketLimiter` takes constructor args rather than reading module
constants directly, so tests can instantiate a small, fast one instead of
sharing process-global state with every other test that hits `/api/ask`
(CLAUDE.md: prefer constructor injection over patching).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, Request

from rag_eval.common.config import settings
from rag_eval.common.telemetry import DEFAULT_LOG_DIR, read_stats

MAX_REQUESTS = 10
WINDOW_SECONDS = 300.0  # 5 minutes

DEMO_BUDGET_EXHAUSTED_DETAIL = (
    "This demo's daily budget is exhausted -- try again tomorrow, or run this project "
    "locally with your own API key (see README)."
)


class TokenBucketLimiter:
    def __init__(self, max_requests: int = MAX_REQUESTS, window_seconds: float = WINDOW_SECONDS) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, now: float | None = None) -> None:
        """Raises HTTPException(429) if `key` has already made
        `max_requests` requests within the trailing window; otherwise
        records this one."""
        now = now if now is not None else time.monotonic()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded -- try again in a few minutes")
        bucket.append(now)


# Module-level singleton -- the actual FastAPI dependency (`rate_limit`
# below) needs one shared limiter across requests; per-test isolation goes
# through overriding that dependency instead (tests/integration/test_ask_route.py).
_limiter = TokenBucketLimiter()


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    _limiter.check(ip)


def _daily_budget_exceeded(log_dir: Path = DEFAULT_LOG_DIR, *, now: datetime | None = None) -> bool:
    return read_stats(log_dir=log_dir, days=1, now=now).total_cost_usd >= settings.daily_budget_usd


def check_daily_budget() -> None:
    """Raises HTTPException(402) once today's logged request cost
    (common/telemetry.py) reaches settings.daily_budget_usd. Takes no
    arguments -- used directly as a FastAPI dependency (`Depends`), and a
    dependency's own parameters become client-facing query params unless
    there are none; the (log_dir-parameterized) check itself is
    `_daily_budget_exceeded`, called here with the real default so tests
    can exercise that function directly instead."""
    if _daily_budget_exceeded():
        raise HTTPException(status_code=402, detail=DEMO_BUDGET_EXHAUSTED_DETAIL)
