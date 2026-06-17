# production-rag-pipeline-v3

**Project 3 of 5 — Monitoring & Observability**

Extends `production-rag-pipeline-v2` (hybrid BM25 + vector retrieval with cross-encoder reranking) with production-grade observability:

- **SQLite metrics store** — p50/p95 latency, cost-per-request, quality trending, citation rate
- **Per-node timing** — every LangGraph node is wrapped with a `timed_node` decorator; timings surface in the API response and metrics DB
- **Langfuse LLM tracing** — optional; set env vars to capture every prompt/completion in the Langfuse dashboard
- **CI regression gates** — GitHub Actions checks p95 ≤ 8000 ms, avg quality ≥ 5.0, citation rate ≥ 50%
- **`/metrics/` API endpoints** — aggregated stats, per-request history, health gate pass/fail

---

## Architecture

```
docs/
 └─ ingest.py                   # chunk → embed → persist (Chroma + BM25)

app/
 ├─ config.py                   # pydantic-settings (env vars + CI thresholds)
 ├─ graph.py                    # LangGraph StateGraph, timed_node decorator
 ├─ api.py                      # FastAPI: /chat/, /metrics/, /metrics/health
 ├─ metrics.py                  # SQLite store — save, query, percentiles, gates
 └─ tracer.py                   # Langfuse handler + TokenCountCallback

tests/
 ├─ conftest.py                 # shared fixtures (in-memory store, mock retriever)
 ├─ test_metrics.py             # 19 regression gate tests (in-memory SQLite)
 └─ test_eval.py                # citation format tests (regex-based)

evaluate.py                     # offline eval: runs 10 Qs, prints gate summary
.github/workflows/ci.yml        # pytest on push/PR + regression gate check
```

### Per-request data collected

| Field | Description |
|-------|-------------|
| `trace_id` | UUID per request |
| `session_id` | Caller-supplied session |
| `total_latency_ms` | End-to-end wall time |
| `node_timings` | Per-node ms: retrieve, rerank, grade, generate, … |
| `input_tokens` | LLM prompt tokens (all calls summed) |
| `output_tokens` | LLM completion tokens |
| `cost_usd` | Estimated cost (Groq llama-3.1-8b-instant rates) |
| `quality_score` | LLM self-grade 1–10 |
| `has_citations` | Boolean — does answer contain `[Source: …]`? |
| `retry_count` | How many retrieval retries were needed |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/akabhishek-kumar/production-rag-pipeline-v3
cd production-rag-pipeline-v3
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in GROQ_API_KEY (required)
# Fill in LANGFUSE_* keys (optional — leave blank to disable)
```

### 3. Ingest documents

```bash
mkdir docs
# Copy your .pdf, .docx, .txt files into docs/
python ingest.py
```

### 4. Run the API

```bash
uvicorn app.api:app --reload
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph?"}'
```

Response:

```json
{
  "answer": "LangGraph is a framework … [Source: langgraph_guide.txt].",
  "session_id": "default",
  "trace_id": "b4f3…",
  "sources": ["langgraph_guide.txt"],
  "latency_ms": 1243.5,
  "cost_usd": 0.0000412,
  "quality_score": 8
}
```

### 6. Check metrics

```bash
# Aggregated stats
curl http://localhost:8000/metrics/

# CI gate pass/fail
curl http://localhost:8000/metrics/health

# Raw per-request data
curl http://localhost:8000/metrics/requests
```

---

## Langfuse setup (optional)

1. Sign up free at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a project → copy public and secret keys
3. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-…
   LANGFUSE_SECRET_KEY=sk-lf-…
   ```
4. Each `/chat/` call will auto-appear in the Langfuse dashboard with full prompt/completion traces

---

## Offline evaluation

```bash
python evaluate.py                # all 10 questions
python evaluate.py --questions 3  # quick smoke test
```

Prints p50/p95 latency, quality, citation rate, and gate pass/fail. Writes `eval_results.json` for CI artifact upload.

---

## CI / regression gates

GitHub Actions runs on every push and PR:

```
pytest tests/ -v --tb=short
```

Gate thresholds (configurable via env vars):

| Gate | Default threshold |
|------|------------------|
| p95 latency | ≤ 8000 ms |
| avg quality | ≥ 5.0 |
| citation rate | ≥ 50% |

All 19 tests use in-memory SQLite — no API key needed, no external services.

---

## Cost model

Groq `llama-3.1-8b-instant` pricing (as of mid-2025):

| | Rate |
|--|--|
| Input tokens | $0.05 / 1M |
| Output tokens | $0.08 / 1M |

Typical request (500 input + 150 output tokens) ≈ **$0.000037**.

---

## Project series

| # | Repo | Focus |
|---|------|-------|
| 1 | production-rag-pipeline | Basic RAG — ingest, retrieve, generate |
| 2 | production-rag-pipeline-v2 | Hybrid BM25 + vector, reranking, self-evaluation, guardrails |
| **3** | **production-rag-pipeline-v3** | **Monitoring & Observability (this repo)** |
| 4 | production-rag-pipeline-v4 | Streaming, async, multi-tenancy |
| 5 | production-rag-pipeline-v5 | Deployment: Docker, k8s, auto-scaling |
