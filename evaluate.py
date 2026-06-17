"""
Offline evaluation script for production-rag-pipeline-v3.

Runs a fixed question set against the live RAG pipeline, persists results to
the metrics store, then prints a regression-gate summary.

Usage:
    python evaluate.py [--questions N] [--db PATH]

Prerequisites:
    1. Ingest docs first:   python ingest.py
    2. Set env vars:        cp .env.example .env  &&  fill in GROQ_API_KEY
"""

import argparse
import json
import time
import uuid
from pathlib import Path

from app.config import Settings
from app.graph import chat
from app.metrics import MetricsStore, RequestMetrics, estimate_cost, now_utc
from app.tracer import TokenCountCallback

EVAL_QUESTIONS = [
    "What is LangGraph and how does it differ from LangChain?",
    "Explain the Harness Engineering pattern for tool registration.",
    "What is RAG and why is retrieval important in LLM pipelines?",
    "How does BM25 differ from vector search?",
    "What is reciprocal rank fusion and when would you use it?",
    "Describe the role of a cross-encoder reranker.",
    "What are the main components of a production RAG pipeline?",
    "How do you evaluate answer quality in a RAG system?",
    "What is a citation and how does it indicate answer grounding?",
    "What observability signals matter most for a RAG pipeline?",
]


def run_eval(n_questions: int, db_path: str) -> None:
    settings = Settings()
    store = MetricsStore(db_path=db_path)
    questions = EVAL_QUESTIONS[:n_questions]

    print(f"\n{'='*60}")
    print(f"  Evaluation — {len(questions)} questions  |  db: {db_path}")
    print(f"{'='*60}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q[:70]}...")
        token_cb = TokenCountCallback()
        t0 = time.perf_counter()

        try:
            answer, final_state = chat(
                question=q,
                session_id="eval",
                extra_callbacks=[token_cb],
            )
            total_ms = (time.perf_counter() - t0) * 1000
            quality = final_state.get("quality_score", 0)
            has_citations = final_state.get("has_citations", False)
            node_timings = final_state.get("_node_timings", {})

            cost = estimate_cost(token_cb.input_tokens, token_cb.output_tokens)
            m = RequestMetrics(
                trace_id=str(uuid.uuid4()),
                session_id="eval",
                question=q,
                answer=answer,
                node_timings=node_timings,
                total_latency_ms=round(total_ms, 2),
                input_tokens=token_cb.input_tokens,
                output_tokens=token_cb.output_tokens,
                cost_usd=cost,
                quality_score=quality,
                has_citations=has_citations,
                retry_count=final_state.get("retry_count", 0),
                timestamp=now_utc(),
            )
            store.save(m)
            results.append(m)
            status = "✓" if quality >= settings.grade_threshold else "✗"
            cite = "[cited]" if has_citations else "[no cite]"
            print(f"  {status} quality={quality}  latency={total_ms:.0f}ms  {cite}\n")

        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            total_ms = (time.perf_counter() - t0) * 1000

    # ── Aggregate stats ────────────────────────────────────────────────────────
    stats = store.compute_stats(n=len(results))
    gates = store.regression_check(
        n=len(results),
        max_p95_ms=settings.ci_max_p95_latency_ms,
        min_quality=settings.ci_min_avg_quality,
        min_citation_rate=settings.ci_min_citation_rate,
    )

    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")

    if "error" in stats:
        print(f"  No data: {stats['error']}")
        return

    print(f"  Questions answered : {len(results)}")
    print(f"  p50 latency        : {stats.get('p50_latency_ms', 0):.0f} ms")
    print(f"  p95 latency        : {stats.get('p95_latency_ms', 0):.0f} ms")
    print(f"  Avg quality        : {stats.get('avg_quality', 0):.2f}")
    print(f"  Citation rate      : {stats.get('citation_rate', 0)*100:.1f}%")
    print(f"  Avg cost / request : ${stats.get('avg_cost_usd', 0):.6f}")

    print(f"\n{'─'*60}")
    print("  REGRESSION GATES")
    print(f"{'─'*60}")

    if gates.get("skipped"):
        print("  SKIPPED — no data in store")
        return

    for key, result in gates.items():
        if key in ("all_pass", "skipped"):
            continue
        icon = "✓ PASS" if result["pass"] else "✗ FAIL"
        print(f"  {icon}  {key}: {result.get('value', '')}  (threshold: {result.get('threshold', '')})")

    overall = "✓ ALL GATES PASS" if gates.get("all_pass") else "✗ ONE OR MORE GATES FAILED"
    print(f"\n  {overall}")
    print(f"{'='*60}\n")

    # Write JSON summary for CI artifact upload
    out = Path("eval_results.json")
    out.write_text(json.dumps({"stats": stats, "gates": gates}, indent=2))
    print(f"  Full results written to {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline v3")
    parser.add_argument("--questions", type=int, default=10, help="Number of eval questions (1-10)")
    parser.add_argument("--db", type=str, default="./eval_metrics.db", help="SQLite DB path for eval run")
    args = parser.parse_args()

    n = max(1, min(args.questions, len(EVAL_QUESTIONS)))
    run_eval(n_questions=n, db_path=args.db)
