from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from rag_eval.api.rate_limit import TokenBucketLimiter, _daily_budget_exceeded
from rag_eval.common.config import settings
from rag_eval.common.telemetry import Usage, log_request


def _usage(**overrides) -> Usage:
    defaults = {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "latency_ms": 250.0,
    }
    return Usage(**{**defaults, **overrides})


def test_token_bucket_allows_up_to_the_limit():
    limiter = TokenBucketLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("1.2.3.4")  # should not raise


def test_token_bucket_blocks_once_over_the_limit():
    limiter = TokenBucketLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("1.2.3.4")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("1.2.3.4")
    assert exc_info.value.status_code == 429


def test_token_bucket_tracks_keys_independently():
    limiter = TokenBucketLimiter(max_requests=1, window_seconds=60)
    limiter.check("a")
    limiter.check("b")  # different key, should not raise

    with pytest.raises(HTTPException):
        limiter.check("a")


def test_token_bucket_forgets_hits_outside_the_window():
    limiter = TokenBucketLimiter(max_requests=1, window_seconds=60)
    limiter.check("a", now=0.0)

    limiter.check("a", now=61.0)  # 61s later, outside the 60s window -- allowed again


def test_daily_budget_not_exceeded_with_no_traffic(tmp_path):
    assert _daily_budget_exceeded(log_dir=tmp_path) is False


def test_daily_budget_exceeded_once_logged_cost_reaches_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "daily_budget_usd", 0.5)
    now = datetime(2026, 8, 30, tzinfo=UTC)
    # groq/openai/gpt-oss-120b: $0.15 in + $0.60 out per 1M tokens
    log_request(
        _usage(prompt_tokens=1_000_000, completion_tokens=1_000_000),
        request_id="r1",
        endpoint="/api/ask",
        log_dir=tmp_path,
        now=now,
    )

    assert _daily_budget_exceeded(log_dir=tmp_path, now=now) is True
