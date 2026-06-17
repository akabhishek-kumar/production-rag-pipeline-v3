"""Shared test fixtures for production RAG pipeline v3."""

import uuid
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.metrics import MetricsStore, RequestMetrics, estimate_cost, now_utc


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="LangGraph is a framework for building stateful multi-actor applications with LLMs.",
            metadata={"source": "langgraph_guide.txt", "chunk_id": 0},
        ),
        Document(
            page_content="RAG stands for Retrieval Augmented Generation.",
            metadata={"source": "rag_intro.txt", "chunk_id": 0},
        ),
        Document(
            page_content="Harness Engineering patterns include Tool Registry and Guardrails.",
            metadata={"source": "harness.txt", "chunk_id": 0},
        ),
    ]


@pytest.fixture
def mock_retriever(sample_docs):
    retriever = MagicMock()
    retriever.invoke.return_value = sample_docs
    return retriever


@pytest.fixture
def in_memory_store() -> MetricsStore:
    """Isolated in-memory SQLite store for each test."""
    return MetricsStore(db_path=":memory:")


def make_test_metric(**kwargs) -> RequestMetrics:
    defaults = dict(
        trace_id=str(uuid.uuid4()),
        session_id="test",
        question="What is RAG?",
        answer="RAG is retrieval augmented generation [Source: rag_intro.txt].",
        node_timings={"retrieve": 400.0, "generate": 800.0},
        total_latency_ms=1200.0,
        input_tokens=500,
        output_tokens=150,
        cost_usd=estimate_cost(500, 150),
        quality_score=7,
        has_citations=True,
        retry_count=0,
        timestamp=now_utc(),
    )
    defaults.update(kwargs)
    return RequestMetrics(**defaults)
