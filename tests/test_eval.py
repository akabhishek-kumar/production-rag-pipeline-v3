"""Unit tests for production RAG pipeline v3.

All tests use mocks — no Groq API key, Chroma DB, or model downloads required.
Safe to run in CI.
"""

import re
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ── Context formatting ────────────────────────────────────────────────────────

def test_format_context_includes_source_headers(sample_docs):
    """_format_context adds [Source: filename, Chunk N] headers."""
    from app.graph import _format_context

    context = _format_context(sample_docs)

    assert "[Source: langgraph_guide.txt, Chunk 0]" in context
    assert "[Source: rag_intro.txt, Chunk 0]" in context
    assert "[Source: harness.txt, Chunk 0]" in context
    assert "LangGraph" in context
    assert "RAG" in context


def test_format_context_empty_docs():
    """_format_context handles empty list gracefully."""
    from app.graph import _format_context

    assert _format_context([]) == ""


# ── Ingestion metadata ────────────────────────────────────────────────────────

def test_split_and_tag_adds_metadata():
    """split_and_tag stamps source (basename) and chunk_id on every chunk."""
    from ingest import split_and_tag

    docs = [
        Document(
            page_content="This is a test document. " * 30,
            metadata={"source": "/some/path/my_file.txt"},
        )
    ]
    chunks = split_and_tag(docs)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["source"] == "my_file.txt"
        assert "chunk_id" in chunk.metadata
        assert isinstance(chunk.metadata["chunk_id"], int)


def test_split_and_tag_chunk_ids_sequential():
    """chunk_id is sequential (0, 1, 2...) within each source."""
    from ingest import split_and_tag

    docs = [
        Document(
            page_content="Paragraph one. " * 40,
            metadata={"source": "doc_a.txt"},
        )
    ]
    chunks = split_and_tag(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


# ── Reranking ─────────────────────────────────────────────────────────────────

def test_rerank_returns_at_most_reranker_k(sample_docs):
    """rerank_documents returns at most reranker_k docs, highest score first."""
    from app.vectorstore import rerank_documents
    from app.config import settings

    with patch("app.vectorstore.get_reranker") as mock_get:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.5, 0.9, 0.3]
        mock_get.return_value = mock_reranker

        result = rerank_documents("test query", sample_docs)

    assert len(result) <= settings.reranker_k
    assert result[0].metadata["reranker_score"] == 0.9


def test_rerank_attaches_score_to_metadata(sample_docs):
    """rerank_documents attaches reranker_score to each doc's metadata."""
    from app.vectorstore import rerank_documents

    scores = [0.7, 0.4, 0.85]
    with patch("app.vectorstore.get_reranker") as mock_get:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = scores
        mock_get.return_value = mock_reranker

        result = rerank_documents("query", sample_docs)

    for doc in result:
        assert "reranker_score" in doc.metadata


def test_rerank_empty_docs():
    """rerank_documents handles empty list without error."""
    from app.vectorstore import rerank_documents

    result = rerank_documents("query", [])
    assert result == []


# ── Citation check ────────────────────────────────────────────────────────────

def test_citation_check_detects_present_citation():
    """[Source: X] pattern is present in a cited answer."""
    answer = "LangGraph is a framework [Source: langgraph_guide.txt]."
    matches = re.findall(r"\[Source:\s*([^\],]+)", answer)
    assert len(matches) > 0, "Expected at least one [Source:] citation"


def test_citation_check_detects_missing_citation():
    """[Source: X] pattern is absent when answer has no citations."""
    answer = "LangGraph is a framework."
    matches = re.findall(r"\[Source:\s*([^\],]+)", answer)
    assert len(matches) == 0, "Expected no [Source:] citations"


# ── API source extraction ─────────────────────────────────────────────────────

def test_extract_sources_single():
    """_extract_sources pulls one source name from answer."""
    from app.api import _extract_sources

    answer = "LangGraph is great [Source: langgraph_guide.txt]."
    assert _extract_sources(answer) == ["langgraph_guide.txt"]


def test_extract_sources_multiple_unique():
    """_extract_sources returns unique sources in order of appearance."""
    from app.api import _extract_sources

    answer = (
        "RAG is useful [Source: rag_intro.txt]. "
        "Harness patterns help [Source: harness.txt]. "
        "RAG again [Source: rag_intro.txt]."
    )
    assert _extract_sources(answer) == ["rag_intro.txt", "harness.txt"]


def test_extract_sources_none():
    """_extract_sources returns empty list when no citations present."""
    from app.api import _extract_sources

    assert _extract_sources("This answer has no citations.") == []


# ── timed_node decorator ──────────────────────────────────────────────────────

def test_timed_node_adds_node_timings_key():
    """timed_node wraps a function and injects _node_timings into result."""
    from app.graph import timed_node

    @timed_node("test_node")
    def dummy(state):
        return {"some_key": "some_value"}

    result = dummy({"_node_timings": {}})
    assert "_node_timings" in result
    assert "test_node" in result["_node_timings"]
    assert result["_node_timings"]["test_node"] > 0


def test_timed_node_accumulates_timings():
    """timed_node preserves timings from previous nodes."""
    from app.graph import timed_node

    @timed_node("second_node")
    def dummy(state):
        return {}

    existing = {"first_node": 300.0}
    result = dummy({"_node_timings": existing})
    assert "first_node" in result["_node_timings"]
    assert "second_node" in result["_node_timings"]
