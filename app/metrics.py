"""SQLite-backed metrics store for production RAG pipeline v3.

Captures per-request observability data:
  - End-to-end latency + per-node breakdown
  - Token counts (input + output) → cost estimation
  - Quality score (1-10 from grade_chain)
  - Citation presence
  - Retry count

Aggregation methods compute p50/p95 latency, cost totals, quality trends,
and citation rates over the last N requests — used by /metrics/ endpoint
and CI regression gates.

Groq llama-3.1-8b-instant pricing (as of 2025):
  Input:  $0.05 per 1M tokens
  Output: $0.08 per 1M tokens
"""

import json
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Groq llama-3.1-8b-instant pricing
_COST_PER_1M_INPUT = 0.05
_COST_PER_1M_OUTPUT = 0.08


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for one request given token counts."""
    return (
        input_tokens * _COST_PER_1M_INPUT + output_tokens * _COST_PER_1M_OUTPUT
    ) / 1_000_000


@dataclass
class RequestMetrics:
    trace_id: str
    session_id: str
    question: str
    answer: str
    node_timings: dict          # {"retrieve": 420.1, "generate": 1340.5, ...}
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: int          # 0 = unknown / pipeline exited early
    has_citations: bool
    retry_count: int
    timestamp: str              # ISO-8601 UTC


class MetricsStore:
    """Thread-safe SQLite metrics store.

    Usage:
        store = MetricsStore()          # opens/creates metrics.db
        store.save(metrics)             # persist one request
        stats = store.compute_stats()   # get aggregated numbers
    """

    def __init__(self, db_path: str = "./metrics.db"):
        self.db_path = db_path
        self._init_db()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_metrics (
                    trace_id         TEXT PRIMARY KEY,
                    session_id       TEXT,
                    question         TEXT,
                    answer           TEXT,
                    node_timings     TEXT,
                    total_latency_ms REAL,
                    input_tokens     INTEGER,
                    output_tokens    INTEGER,
                    cost_usd         REAL,
                    quality_score    INTEGER,
                    has_citations    INTEGER,
                    retry_count      INTEGER,
                    timestamp        TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, m: RequestMetrics) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO request_metrics VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    m.trace_id,
                    m.session_id,
                    m.question,
                    m.answer,
                    json.dumps(m.node_timings),
                    m.total_latency_ms,
                    m.input_tokens,
                    m.output_tokens,
                    m.cost_usd,
                    m.quality_score,
                    int(m.has_citations),
                    m.retry_count,
                    m.timestamp,
                ),
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_recent(self, n: int = 100) -> list[RequestMetrics]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM request_metrics ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self._row_to_metrics(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM request_metrics"
            ).fetchone()[0]

    # ── Aggregation ───────────────────────────────────────────────────────────

    def compute_stats(self, n: int = 100) -> dict:
        """Return aggregated stats over the last N requests.

        Returns a dict suitable for JSON serialization:
        {
            "total_requests": 47,
            "p50_latency_ms": 1240.5,
            "p95_latency_ms": 3812.0,
            "avg_cost_usd": 0.000031,
            "total_cost_usd": 0.0015,
            "avg_quality_score": 7.2,
            "citation_rate": 0.94,
            "avg_retry_count": 0.21,
            "avg_node_latency_ms": {"retrieve": 410.3, "generate": 1180.2, ...},
            "window": 100,
        }
        """
        metrics = self.get_recent(n)
        if not metrics:
            return {"error": "no data yet — make some /chat/ requests first"}

        latencies = [m.total_latency_ms for m in metrics]
        costs = [m.cost_usd for m in metrics]
        quality = [m.quality_score for m in metrics if m.quality_score > 0]
        citations = [m.has_citations for m in metrics]
        retries = [m.retry_count for m in metrics]

        sorted_lat = sorted(latencies)
        p50 = statistics.median(sorted_lat)
        p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        p95 = sorted_lat[p95_idx]

        # Per-node average timing
        node_sums: dict[str, list[float]] = {}
        for m in metrics:
            for node, ms in m.node_timings.items():
                node_sums.setdefault(node, []).append(ms)
        avg_node = {k: round(statistics.mean(v), 1) for k, v in node_sums.items()}

        return {
            "window": n,
            "total_requests": len(metrics),
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "avg_cost_usd": round(statistics.mean(costs), 7),
            "total_cost_usd": round(sum(costs), 5),
            "avg_quality_score": round(statistics.mean(quality), 2) if quality else 0,
            "citation_rate": round(sum(citations) / len(citations), 3),
            "avg_retry_count": round(statistics.mean(retries), 2),
            "avg_node_latency_ms": avg_node,
        }

    def regression_check(
        self,
        n: int = 100,
        max_p95_ms: float = 8000.0,
        min_quality: float = 5.0,
        min_citation_rate: float = 0.5,
    ) -> dict:
        """Return pass/fail for each CI gate threshold.

        Used by tests/test_metrics.py to gate CI on quality regression.
        """
        stats = self.compute_stats(n)
        if "error" in stats:
            return {"skipped": True, "reason": stats["error"]}

        gates = {
            "p95_latency": {
                "value": stats["p95_latency_ms"],
                "threshold": max_p95_ms,
                "pass": stats["p95_latency_ms"] <= max_p95_ms,
                "label": f"p95 ≤ {max_p95_ms}ms",
            },
            "avg_quality": {
                "value": stats["avg_quality_score"],
                "threshold": min_quality,
                "pass": stats["avg_quality_score"] >= min_quality,
                "label": f"avg quality ≥ {min_quality}",
            },
            "citation_rate": {
                "value": stats["citation_rate"],
                "threshold": min_citation_rate,
                "pass": stats["citation_rate"] >= min_citation_rate,
                "label": f"citation rate ≥ {min_citation_rate:.0%}",
            },
        }
        gates["all_pass"] = all(g["pass"] for g in gates.values())
        return gates

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_metrics(row: tuple) -> RequestMetrics:
        return RequestMetrics(
            trace_id=row[0],
            session_id=row[1],
            question=row[2],
            answer=row[3],
            node_timings=json.loads(row[4]),
            total_latency_ms=row[5],
            input_tokens=row[6],
            output_tokens=row[7],
            cost_usd=row[8],
            quality_score=row[9],
            has_citations=bool(row[10]),
            retry_count=row[11],
            timestamp=row[12],
        )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
