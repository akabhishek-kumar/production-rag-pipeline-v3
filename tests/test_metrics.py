"""Regression gate tests for production RAG pipeline v3.

All tests use in-memory SQLite (db_path=":memory:") — no file created,
no API key needed, no Chroma DB required. Safe for CI.

CI gates (seeded with realistic test data):
  p95 latency     ≤ 8000ms
  avg quality     ≥ 5.0
  citation rate   ≥ 50%

The seeded data represents a realistic distribution:
  - Mix of fast (800ms) and slow (4500ms) requests
  - Quality scores 6-9 (passing threshold)
  - 80% citation rate
"""

import uuid
from datetime import datetime, timezone

import pytest

from app.metrics import MetricsStore, RequestMetrics, estimate_cost, now_utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_metric(
    latency_ms: float = 1200.0,
    quality_score: int = 7,
    has_citations: bool = True,
    retry_count: int = 0,
    input_tokens: int = 500,
    output_tokens: int = 150,
    node_timings: dict | None = None,
) -> RequestMetrics:
    return RequestMetrics(
        trace_id=str(uuid.uuid4()),
        session_id="test-session",
        question="What is LangGraph?",
        answer="LangGraph is a framework [Source: langgraph_guide.txt].",
        node_timings=node_timings or {"retrieve": 400.0, "generate": 700.0},
        total_latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate_cost(input_tokens, output_tokens),
        quality_score=quality_score,
        has_citations=has_citations,
        retry_count=retry_count,
        timestamp=now_utc(),
    )


@pytest.fixture
def store() -> MetricsStore:
    """In-memory SQLite store — isolated per test."""
    return MetricsStore(db_path=":memory:")


@pytest.fixture
def seeded_store() -> MetricsStore:
    """Store pre-loaded with 20 realistic requests."""
    s = MetricsStore(db_path=":memory:")
    # 16 fast, high-quality requests with citations
    for _ in range(16):
        s.save(_make_metric(latency_ms=1100.0, quality_score=7, has_citations=True))
    # 2 slow requests (but still within p95 gate)
    for _ in range(2):
        s.save(_make_metric(latency_ms=4500.0, quality_score=6, has_citations=True))
    # 2 requests without citations (20% — still above 50% gate)
    for _ in range(2):
        s.save(_make_metric(latency_ms=1300.0, quality_score=6, has_citations=False))
    return s


# ── MetricsStore unit tests ───────────────────────────────────────────────────

def test_store_save_and_retrieve(store):
    """Saved metrics can be retrieved."""
    m = _make_metric()
    store.save(m)
    recent = store.get_recent(10)
    assert len(recent) == 1
    assert recent[0].trace_id == m.trace_id


def test_store_count(store):
    """count() returns number of saved records."""
    assert store.count() == 0
    store.save(_make_metric())
    store.save(_make_metric())
    assert store.count() == 2


def test_store_get_recent_limit(store):
    """get_recent(n) returns at most n records."""
    for _ in range(10):
        store.save(_make_metric())
    assert len(store.get_recent(5)) == 5


def test_store_preserves_node_timings(store):
    """node_timings dict is serialized and deserialized correctly."""
    timings = {"retrieve": 412.5, "generate": 1340.2, "evaluate": 230.1}
    m = _make_metric(node_timings=timings)
    store.save(m)
    retrieved = store.get_recent(1)[0]
    assert retrieved.node_timings == timings


def test_store_no_data_returns_error(store):
    """compute_stats() returns error dict when no data exists."""
    result = store.compute_stats()
    assert "error" in result


# ── Aggregation / percentile tests ────────────────────────────────────────────

def test_p50_latency(store):
    """p50 is the median latency."""
    latencies = [500.0, 1000.0, 1500.0, 2000.0, 2500.0]
    for lat in latencies:
        store.save(_make_metric(latency_ms=lat))
    stats = store.compute_stats()
    assert stats["p50_latency_ms"] == 1500.0


def test_p95_latency(store):
    """p95 is near the 95th percentile of latencies."""
    # 20 requests: 19 fast (1000ms) + 1 very slow (9000ms)
    for _ in range(19):
        store.save(_make_metric(latency_ms=1000.0))
    store.save(_make_metric(latency_ms=9000.0))
    stats = store.compute_stats()
    # p95 should capture the slow outlier
    assert stats["p95_latency_ms"] >= 9000.0


def test_citation_rate(store):
    """citation_rate is fraction of requests with citations."""
    store.save(_make_metric(has_citations=True))
    store.save(_make_metric(has_citations=True))
    store.save(_make_metric(has_citations=False))
    stats = store.compute_stats()
    assert abs(stats["citation_rate"] - 2 / 3) < 0.01


def test_avg_node_latency(store):
    """avg_node_latency_ms is computed per node across requests."""
    store.save(_make_metric(node_timings={"retrieve": 400.0, "generate": 800.0}))
    store.save(_make_metric(node_timings={"retrieve": 600.0, "generate": 1200.0}))
    stats = store.compute_stats()
    assert stats["avg_node_latency_ms"]["retrieve"] == 500.0
    assert stats["avg_node_latency_ms"]["generate"] == 1000.0


# ── Cost estimation ───────────────────────────────────────────────────────────

def test_estimate_cost_zero_tokens():
    assert estimate_cost(0, 0) == 0.0


def test_estimate_cost_known_values():
    """1M input + 1M output tokens = $0.05 + $0.08 = $0.13."""
    cost = estimate_cost(1_000_000, 1_000_000)
    assert abs(cost - 0.13) < 0.0001


def test_estimate_cost_small_request():
    """Typical request: 500 input + 150 output tokens."""
    cost = estimate_cost(500, 150)
    assert cost > 0
    assert cost < 0.001   # should be fractions of a cent


# ── CI regression gate tests ──────────────────────────────────────────────────

def test_ci_gates_pass_on_good_data(seeded_store):
    """All gates pass when metrics are healthy."""
    result = seeded_store.regression_check(
        max_p95_ms=8000.0,
        min_quality=5.0,
        min_citation_rate=0.5,
    )
    assert result.get("all_pass") is True, f"Gates failed: {result}"


def test_ci_gate_p95_fails_on_slow_requests(store):
    """p95 gate fails when requests are consistently slow."""
    for _ in range(20):
        store.save(_make_metric(latency_ms=10000.0))
    result = store.regression_check(max_p95_ms=8000.0)
    assert result["p95_latency"]["pass"] is False


def test_ci_gate_quality_fails_on_low_scores(store):
    """Quality gate fails when avg score is below threshold."""
    for _ in range(10):
        store.save(_make_metric(quality_score=3))
    result = store.regression_check(min_quality=5.0)
    assert result["avg_quality"]["pass"] is False


def test_ci_gate_citations_fails_on_low_rate(store):
    """Citation gate fails when fewer than 50% of answers cite sources."""
    for _ in range(8):
        store.save(_make_metric(has_citations=False))
    for _ in range(2):
        store.save(_make_metric(has_citations=True))
    result = store.regression_check(min_citation_rate=0.5)
    assert result["citation_rate"]["pass"] is False


def test_ci_gates_skip_when_no_data(store):
    """regression_check returns skipped=True when store is empty."""
    result = store.regression_check()
    assert result.get("skipped") is True
