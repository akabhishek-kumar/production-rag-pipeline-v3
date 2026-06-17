"""FastAPI application — production RAG v3.

v3 vs v2 additions:
  - Every /chat/ request is timed end-to-end
  - TokenCountCallback captures input/output tokens → cost estimation
  - Optional Langfuse handler injected for LLM tracing
  - RequestMetrics saved to SQLite after each request
  - GET /metrics/          → aggregated stats (p50/p95, cost, quality, citations)
  - GET /metrics/health    → CI gate pass/fail
  - GET /metrics/requests  → last N raw requests
"""

import re
import time
import uuid

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import chat
from app.metrics import MetricsStore, RequestMetrics, estimate_cost, now_utc
from app.tracer import TokenCountCallback, get_langfuse_handler

app = FastAPI(
    title="Production RAG Pipeline v3",
    description="RAG with hybrid retrieval, reranking, citations, and full observability.",
    version="3.0.0",
)

# Singleton metrics store — shared across requests
_store = MetricsStore(settings.metrics_db_path)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    trace_id: str
    sources: list[str] = Field(default=[], description="Cited source filenames")
    latency_ms: float = Field(description="End-to-end request latency in ms")
    cost_usd: float = Field(description="Estimated Groq API cost for this request")
    quality_score: int = Field(description="Answer quality grade 1-10 (0 = unknown)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_sources(answer: str) -> list[str]:
    """Parse all unique [Source: filename] tags from the answer."""
    matches = re.findall(r"\[Source:\s*([^\],]+)", answer)
    seen: list[str] = []
    for m in matches:
        name = m.strip()
        if name not in seen:
            seen.append(name)
    return seen


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    trace_id = str(uuid.uuid4())

    # Build callbacks: token counter always on, Langfuse when keys are set
    token_cb = TokenCountCallback()
    callbacks = [token_cb]
    langfuse_handler = get_langfuse_handler(trace_id, req.session_id)
    if langfuse_handler:
        callbacks.append(langfuse_handler)

    # Run the pipeline
    t0 = time.perf_counter()
    answer, final_state = chat(
        question=req.question,
        session_id=req.session_id,
        extra_callbacks=callbacks,
    )
    total_ms = (time.perf_counter() - t0) * 1000

    # Cost estimation
    cost = estimate_cost(token_cb.input_tokens, token_cb.output_tokens)

    # Persist metrics
    _store.save(RequestMetrics(
        trace_id=trace_id,
        session_id=req.session_id,
        question=req.question,
        answer=answer,
        node_timings=final_state.get("_node_timings", {}),
        total_latency_ms=round(total_ms, 2),
        input_tokens=token_cb.input_tokens,
        output_tokens=token_cb.output_tokens,
        cost_usd=cost,
        quality_score=final_state.get("quality_score", 0),
        has_citations=final_state.get("has_citations", False),
        retry_count=final_state.get("retry_count", 0),
        timestamp=now_utc(),
    ))

    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        trace_id=trace_id,
        sources=_extract_sources(answer),
        latency_ms=round(total_ms, 2),
        cost_usd=round(cost, 7),
        quality_score=final_state.get("quality_score", 0),
    )


# ── Metrics endpoints ─────────────────────────────────────────────────────────

@app.get("/metrics/")
async def get_metrics(window: int = Query(default=100, ge=1, le=1000)):
    """Aggregated observability stats over the last `window` requests.

    Returns:
        p50_latency_ms, p95_latency_ms  — latency percentiles
        avg_cost_usd, total_cost_usd    — cost tracking
        avg_quality_score               — answer quality trend
        citation_rate                   — % of answers with citations
        avg_node_latency_ms             — per-node breakdown
    """
    return _store.compute_stats(n=window)


@app.get("/metrics/health")
async def metrics_health(window: int = Query(default=100, ge=1, le=1000)):
    """CI regression gate — pass/fail for each threshold.

    Thresholds (configurable in .env):
        p95 latency    <= CI_MAX_P95_LATENCY_MS  (default 8000ms)
        avg quality    >= CI_MIN_AVG_QUALITY      (default 5.0)
        citation rate  >= CI_MIN_CITATION_RATE    (default 50%)
    """
    return _store.regression_check(
        n=window,
        max_p95_ms=settings.ci_max_p95_latency_ms,
        min_quality=settings.ci_min_avg_quality,
        min_citation_rate=settings.ci_min_citation_rate,
    )


@app.get("/metrics/requests")
async def get_recent_requests(n: int = Query(default=20, ge=1, le=200)):
    """Raw per-request metrics — last N requests, newest first."""
    metrics = _store.get_recent(n)
    return [
        {
            "trace_id": m.trace_id,
            "timestamp": m.timestamp,
            "question": m.question[:80] + "..." if len(m.question) > 80 else m.question,
            "total_latency_ms": m.total_latency_ms,
            "node_timings": m.node_timings,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "cost_usd": m.cost_usd,
            "quality_score": m.quality_score,
            "has_citations": m.has_citations,
            "retry_count": m.retry_count,
        }
        for m in metrics
    ]


@app.get("/")
async def root():
    return {
        "service": "Production RAG Pipeline v3",
        "endpoints": {
            "chat": "POST /chat/",
            "metrics": "GET /metrics/",
            "health": "GET /metrics/health",
            "requests": "GET /metrics/requests",
            "docs": "GET /docs",
        },
        "total_requests": _store.count(),
    }
