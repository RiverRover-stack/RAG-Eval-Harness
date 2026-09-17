import json
from datetime import UTC, datetime

import pytest

from rag_eval.common.telemetry import (
    Usage,
    estimate_cost,
    log_feedback,
    log_request,
    read_stats,
    timed,
)


def _usage(**overrides) -> Usage:
    defaults = {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "latency_ms": 250.0,
    }
    return Usage(**{**defaults, **overrides})


def test_estimate_cost_known_model():
    usage = _usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    # groq/openai/gpt-oss-120b: $0.15 in + $0.60 out per 1M tokens
    assert estimate_cost(usage) == 0.75


def test_estimate_cost_unknown_model_returns_none():
    usage = _usage(provider="groq", model="some-unlisted-model")
    assert estimate_cost(usage) is None


def test_estimate_cost_local_ollama_is_free():
    usage = _usage(provider="ollama", model="fdm-llama", prompt_tokens=5000, completion_tokens=5000)
    assert estimate_cost(usage) == 0.0


def test_timed_reports_nonnegative_elapsed_and_keeps_increasing():
    with timed() as elapsed_ms:
        first = elapsed_ms()
    second = elapsed_ms()

    assert first >= 0.0
    assert second >= first


def test_log_request_appends_a_json_line_to_the_dated_file(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    usage = _usage()

    path = log_request(usage, request_id="r1", endpoint="/api/ask", log_dir=tmp_path, now=now)

    assert path == tmp_path / "requests-2026-08-30.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["request_id"] == "r1"
    assert row["endpoint"] == "/api/ask"
    assert row["provider"] == "groq"
    assert row["model"] == "openai/gpt-oss-120b"
    assert row["abstained"] is False
    assert row["cost_usd"] == estimate_cost(usage)


def test_log_request_appends_rather_than_overwrites(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    log_request(_usage(), request_id="r1", endpoint="/api/ask", log_dir=tmp_path, now=now)
    path = log_request(_usage(), request_id="r2", endpoint="/api/ask", log_dir=tmp_path, now=now)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["request_id"] == "r2"


def test_log_request_records_abstained_flag(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    path = log_request(
        _usage(), request_id="r1", endpoint="/api/ask", abstained=True, log_dir=tmp_path, now=now
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["abstained"] is True


def test_log_feedback_appends_a_json_line_to_the_dated_file(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)

    path = log_feedback(request_id="r1", verdict="good", log_dir=tmp_path, now=now)

    assert path == tmp_path / "feedback-2026-08-30.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["request_id"] == "r1"
    assert row["verdict"] == "good"


def test_log_feedback_appends_rather_than_overwrites(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    log_feedback(request_id="r1", verdict="good", log_dir=tmp_path, now=now)
    path = log_feedback(request_id="r2", verdict="bad", log_dir=tmp_path, now=now)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["verdict"] == "bad"


def test_read_stats_on_empty_log_dir_is_all_zero(tmp_path):
    stats = read_stats(log_dir=tmp_path)
    assert stats.request_count == 0
    assert stats.total_cost_usd == 0.0
    assert stats.p50_latency_ms is None
    assert stats.p95_latency_ms is None


def test_read_stats_counts_requests_and_sums_cost(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    log_request(_usage(latency_ms=100.0), request_id="r1", endpoint="/api/ask", log_dir=tmp_path, now=now)
    log_request(_usage(latency_ms=200.0), request_id="r2", endpoint="/api/ask", log_dir=tmp_path, now=now)

    stats = read_stats(log_dir=tmp_path, now=now)

    assert stats.request_count == 2
    assert stats.priced_request_count == 2
    assert stats.total_cost_usd == pytest.approx(2 * estimate_cost(_usage()))


def test_read_stats_excludes_unpriced_requests_from_cost_but_counts_them(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    log_request(_usage(), request_id="r1", endpoint="/api/ask", log_dir=tmp_path, now=now)
    log_request(
        _usage(provider="groq", model="unlisted-model"),
        request_id="r2",
        endpoint="/api/ask",
        log_dir=tmp_path,
        now=now,
    )

    stats = read_stats(log_dir=tmp_path, now=now)

    assert stats.request_count == 2
    assert stats.priced_request_count == 1
    assert stats.total_cost_usd == pytest.approx(estimate_cost(_usage()))


def test_read_stats_computes_p50_and_p95_latency(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    for latency in [10.0, 20.0, 30.0, 40.0, 100.0]:
        log_request(_usage(latency_ms=latency), request_id=f"r{latency}", endpoint="/api/ask", log_dir=tmp_path, now=now)

    stats = read_stats(log_dir=tmp_path, now=now)

    assert stats.p50_latency_ms == 30.0
    assert stats.p95_latency_ms == 100.0


def test_read_stats_only_looks_back_the_given_number_of_days(tmp_path):
    now = datetime(2026, 8, 30, tzinfo=UTC)
    old_day = datetime(2026, 8, 1, tzinfo=UTC)
    log_request(_usage(), request_id="old", endpoint="/api/ask", log_dir=tmp_path, now=old_day)
    log_request(_usage(), request_id="new", endpoint="/api/ask", log_dir=tmp_path, now=now)

    stats = read_stats(log_dir=tmp_path, now=now, days=7)

    assert stats.request_count == 1
