"""Lightweight request telemetry (docs/plan.md Phase 7) -- no OpenTelemetry,
just enough to answer "how much did the demo cost" and "how slow is it" from
a JSONL log, appended once per request to `runs/serve/requests-YYYY-MM-DD.jsonl`.

`GET /api/stats` (api/routes/health.py) reads these logs back to report
p50/p95 latency, total cost, and request count.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_LOG_DIR = Path("runs/serve")

# $ per 1M (prompt, completion) tokens. Point-in-time snapshots, not a live
# pricing API -- verify against the provider's own pricing page before
# trusting a cost number for billing, not just for the demo's cost/latency
# panel. Keyed as "{provider}/{model}" so two providers can't collide on a
# bare model name.
#
# llama-3.3-70b-versatile (the model this project's docs/plan.md originally
# locked in as the Groq default) moved to Enterprise-only "Contact Sales"
# pricing on Groq as of 2026-08-26 and was withdrawn from the self-serve
# rate card, taking its public price with it -- the default was moved to
# gpt-oss-120b (config/run_config.py, providers/llm/groq.py) for that reason.
PRICES: dict[str, tuple[float, float]] = {
    "groq/openai/gpt-oss-120b": (0.15, 0.60),
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "ollama/fdm-llama": (0.0, 0.0),  # local inference, no per-token cost
}


@dataclass(frozen=True)
class Usage:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def estimate_cost(usage: Usage) -> float | None:
    """USD cost for one request, or None if the provider/model combination
    has no entry in PRICES -- an unpriced request should show up as
    "unavailable", not as a silently wrong 0.0 or an invented number."""
    key = f"{usage.provider}/{usage.model}"
    prices = PRICES.get(key)
    if prices is None:
        return None
    prompt_price, completion_price = prices
    return (usage.prompt_tokens * prompt_price + usage.completion_tokens * completion_price) / 1_000_000


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    """`with timed() as elapsed_ms: ...; elapsed_ms()` returns milliseconds
    elapsed since the `with` block was entered, callable both inside and
    after the block."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def log_request(
    usage: Usage,
    *,
    request_id: str,
    endpoint: str,
    abstained: bool = False,
    log_dir: Path = DEFAULT_LOG_DIR,
    now: datetime | None = None,
) -> Path:
    """Append one JSON line to today's request log, creating the directory
    and file as needed. Returns the log file path written to."""
    now = now or datetime.now(UTC)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"requests-{now:%Y-%m-%d}.jsonl"

    row = {
        "request_id": request_id,
        "endpoint": endpoint,
        "timestamp": now.isoformat(),
        "abstained": abstained,
        "cost_usd": estimate_cost(usage),
        **asdict(usage),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return log_path


def log_feedback(
    *,
    request_id: str,
    verdict: str,
    log_dir: Path = DEFAULT_LOG_DIR,
    now: datetime | None = None,
) -> Path:
    """Append one JSON line to today's feedback log (POST /api/feedback),
    same append-only pattern as `log_request` -- creates the directory and
    file as needed. Returns the log file path written to."""
    now = now or datetime.now(UTC)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"feedback-{now:%Y-%m-%d}.jsonl"

    row = {"request_id": request_id, "verdict": verdict, "timestamp": now.isoformat()}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return log_path


def log_verdict(
    *,
    request_id: str,
    verdict: str,
    log_dir: Path = DEFAULT_LOG_DIR,
    now: datetime | None = None,
) -> Path:
    """Append one JSON line to today's reviewer-verdict log (POST
    /api/verdict), same append-only pattern as `log_feedback` -- creates the
    directory and file as needed. Kept in its own log file, separate from
    `log_feedback`'s end-user Yes/No, so the two audiences never merge.
    Returns the log file path written to."""
    now = now or datetime.now(UTC)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"verdict-{now:%Y-%m-%d}.jsonl"

    row = {"request_id": request_id, "verdict": verdict, "timestamp": now.isoformat()}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return log_path


@dataclass(frozen=True)
class Stats:
    request_count: int
    priced_request_count: int
    total_cost_usd: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(len(sorted_values) * p))
    return sorted_values[index]


def read_stats(*, log_dir: Path = DEFAULT_LOG_DIR, days: int = 7, now: datetime | None = None) -> Stats:
    """Aggregate the last `days` days of request logs for `GET /api/stats`.
    Missing days (no traffic yet, or a fresh deploy) are simply skipped --
    an empty log directory yields a Stats with zero counts and no
    percentiles, not an error."""
    now = now or datetime.now(UTC)
    rows: list[dict] = []
    for offset in range(days):
        day = now - timedelta(days=offset)
        log_path = log_dir / f"requests-{day:%Y-%m-%d}.jsonl"
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    latencies = sorted(row["latency_ms"] for row in rows)
    costs = [row["cost_usd"] for row in rows if row.get("cost_usd") is not None]
    return Stats(
        request_count=len(rows),
        priced_request_count=len(costs),
        total_cost_usd=sum(costs),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )
