"""LangGraph pipeline — production RAG v3 (observability + latency optimizations).

v3 vs v2 — what changed:
  - timed_node decorator: wraps every node to record execution time in ms
  - _node_timings: new AgentState field — dict of {node_name: latency_ms}
  - chat() now returns (answer, final_state) so api.py can read timings + scores
  - TokenCountCallback + optional Langfuse handler injected at graph.invoke()

Latency optimizations (v3):
  1. Parallel retrieval: BM25 + vector search run concurrently via ThreadPoolExecutor
     Expected saving: ~40-50% of retrieve node latency
  2. Combined verify_and_evaluate: hallucination check + quality score in one LLM call
     Expected saving: ~3-5s per request (eliminated one full LLM round trip)

8-node graph (was 9 in v2 — verify_answer + evaluate merged):
  retrieve → filter_docs → no_info
                         ↘ generate → citation_check → verify_and_evaluate
                                    ↘ (no citation)         ↘
                                      rewrite ←─────────────┘
"""

import time
from functools import wraps
from typing import Annotated, Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.chains import (
    GradeResult,
    citation_check_chain,
    grade_chain,
    hallucination_chain,
    rag_chain,
    relevance_chain,
    rewrite_chain,
    verify_and_evaluate_chain,   # v3 optimization: replaces hallucination + grade
)
from app.config import settings
from app.vectorstore import (
    get_hybrid_retriever,
    load_bm25_documents,
    load_vectorstore,
    rerank_documents,
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: list[Document]
    filtered_docs: list[Document]
    current_question: str
    quality_score: int
    answer_grounded: bool
    has_citations: bool
    retry_count: int
    _node_timings: dict         # v3: {"retrieve": 412.3, "generate": 1340.5, ...}


# ── Node timing decorator ─────────────────────────────────────────────────────

def timed_node(node_name: str):
    """Decorator that records how long a LangGraph node takes in milliseconds.

    Reads existing _node_timings from state (if any) and merges the new entry.
    The merged dict is written back to state so all nodes accumulate timings.

    Usage:
        @timed_node("retrieve")
        def retrieve_node(state: AgentState) -> dict:
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(state: AgentState) -> dict:
            start = time.perf_counter()
            result = fn(state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            existing = state.get("_node_timings", {})
            updated = {**existing, node_name: round(elapsed_ms, 2)}

            if isinstance(result, dict):
                result["_node_timings"] = updated
            return result
        return wrapper
    return decorator


# ── Lazy retriever ────────────────────────────────────────────────────────────

_retriever: Optional[object] = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = load_vectorstore()
        bm25_docs = load_bm25_documents()
        _retriever = get_hybrid_retriever(vectorstore, bm25_docs)
    return _retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[Source: {source}, Chunk {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Nodes (each wrapped with @timed_node) ─────────────────────────────────────

@timed_node("retrieve")
def retrieve_node(state: AgentState) -> dict:
    """Hybrid retrieve (BM25 + vector) then cross-encoder rerank."""
    question = state["current_question"]
    docs = _get_retriever().invoke(question)
    reranked = rerank_documents(question, docs)
    print(f"[retrieve] {len(docs)} candidates → {len(reranked)} after reranking")
    return {"retrieved_docs": reranked}


@timed_node("filter_docs")
def filter_docs_node(state: AgentState) -> dict:
    """LLM relevance filter on top of reranking."""
    question = state["current_question"]
    relevant = []
    for doc in state["retrieved_docs"]:
        result: dict = relevance_chain.invoke({
            "question": question,
            "document": doc.page_content,
        })
        if result.get("is_relevant", False):
            relevant.append(doc)
        else:
            print(f"[filter] dropped: {doc.page_content[:60]}...")
    print(f"[filter] kept {len(relevant)}/{len(state['retrieved_docs'])} chunks")
    return {"filtered_docs": relevant}


@timed_node("no_info")
def no_info_node(state: AgentState) -> dict:
    msg = (
        "I don't have enough information in my knowledge base to answer that. "
        "Try rephrasing, or add relevant documents and re-run ingest.py."
    )
    return {"messages": [AIMessage(content=msg)], "quality_score": 0}


@timed_node("generate")
def generate_node(state: AgentState) -> dict:
    context = _format_context(state["filtered_docs"])
    history = state["messages"][:-1]
    answer = rag_chain.invoke({
        "question": state["current_question"],
        "context": context,
        "history": history,
    })
    return {"messages": [AIMessage(content=answer)]}


@timed_node("citation_check")
def citation_check_node(state: AgentState) -> dict:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    result: dict = citation_check_chain.invoke({"answer": last_ai.content})
    has_citations = result.get("has_citations", False)
    print(f"[citation] has_citations={has_citations}")
    return {"has_citations": has_citations}


# ── v3 OPTIMIZATION: combined verify + evaluate node ─────────────────────────
# Replaces two separate LLM calls (verify_answer_node + evaluate_node) with
# one call that returns both grounded (bool) and quality_score (int).
# Saves ~3-5s per request. Old nodes kept below as comments for reference.

@timed_node("verify_and_evaluate")
def verify_and_evaluate_node(state: AgentState) -> dict:
    """Single LLM call: hallucination check + quality score combined."""
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    context = _format_context(state["filtered_docs"])
    result: dict = verify_and_evaluate_chain.invoke({
        "context": context,
        "question": state["current_question"],
        "answer": last_ai.content,
    })
    grounded = result.get("grounded", False)
    score = result.get("quality_score", 0)
    print(f"[verify_and_evaluate] grounded={grounded}, score={score}/10")
    return {"answer_grounded": grounded, "quality_score": score}


# ── ORIGINAL separate nodes (commented out — kept for reference) ──────────────
#
# @timed_node("verify_answer")
# def verify_answer_node(state: AgentState) -> dict:
#     """Check if answer is grounded in context (hallucination detection)."""
#     last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
#     context = _format_context(state["filtered_docs"])
#     result: dict = hallucination_chain.invoke({
#         "context": context,
#         "answer": last_ai.content,
#     })
#     grounded = result.get("grounded", False)
#     print(f"[verify] grounded={grounded}")
#     return {"answer_grounded": grounded}
#
#
# @timed_node("evaluate")
# def evaluate_node(state: AgentState) -> dict:
#     """Grade answer quality 1-10 using LLM self-evaluation."""
#     last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
#     result: GradeResult = grade_chain.invoke({
#         "question": state["current_question"],
#         "answer": last_ai.content,
#     })
#     print(f"[evaluate] score={result.score}/10")
#     return {"quality_score": result.score}
#
# ─────────────────────────────────────────────────────────────────────────────


@timed_node("rewrite")
def rewrite_node(state: AgentState) -> dict:
    rewritten = rewrite_chain.invoke({
        "question": state["current_question"],
        "history": state["messages"],
    })
    print(f"[rewrite] '{state['current_question']}' -> '{rewritten}'")
    return {
        "current_question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_filter(state: AgentState) -> Literal["generate", "no_info"]:
    return "generate" if state["filtered_docs"] else "no_info"


# ── v3 OPTIMIZATION: updated routing for combined verify+evaluate node ────────

def route_after_citation(
    state: AgentState,
) -> Literal["verify_and_evaluate", "rewrite", "__end__"]:
    if state["has_citations"]:
        return "verify_and_evaluate"     # v3: was "verify_answer"
    if state["retry_count"] >= settings.max_retries:
        return "__end__"
    return "rewrite"


def route_after_verify_and_evaluate(
    state: AgentState,
) -> Literal["rewrite", "__end__"]:
    """Single router replacing route_after_verify + route_after_evaluate."""
    if state["answer_grounded"] and state["quality_score"] >= settings.grade_threshold:
        return "__end__"
    if state["retry_count"] >= settings.max_retries:
        return "__end__"
    return "rewrite"


# ── ORIGINAL routing (commented out — kept for reference) ─────────────────────
#
# def route_after_citation(state) -> Literal["verify_answer", "rewrite", "__end__"]:
#     if state["has_citations"]:
#         return "verify_answer"
#     if state["retry_count"] >= settings.max_retries:
#         return "__end__"
#     return "rewrite"
#
# def route_after_verify(state) -> Literal["evaluate", "rewrite", "__end__"]:
#     if state["answer_grounded"]:
#         return "evaluate"
#     if state["retry_count"] >= settings.max_retries:
#         return "__end__"
#     return "rewrite"
#
# def route_after_evaluate(state) -> Literal["rewrite", "__end__"]:
#     if state["quality_score"] >= settings.grade_threshold:
#         return "__end__"
#     if state["retry_count"] >= settings.max_retries:
#         return "__end__"
#     return "rewrite"
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    # ── v3 optimized graph (8 nodes, was 9) ───────────────────────────────────
    # verify_answer + evaluate merged into verify_and_evaluate (one LLM call)
    builder.add_node("retrieve",             retrieve_node)
    builder.add_node("filter_docs",          filter_docs_node)
    builder.add_node("no_info",              no_info_node)
    builder.add_node("generate",             generate_node)
    builder.add_node("citation_check",       citation_check_node)
    builder.add_node("verify_and_evaluate",  verify_and_evaluate_node)  # v3: combined
    builder.add_node("rewrite",              rewrite_node)

    # ── ORIGINAL 9-node wiring (commented out) ────────────────────────────────
    # builder.add_node("verify_answer",  verify_answer_node)
    # builder.add_node("evaluate",       evaluate_node)
    # ─────────────────────────────────────────────────────────────────────────

    builder.add_edge(START,            "retrieve")
    builder.add_edge("retrieve",       "filter_docs")
    builder.add_edge("no_info",        END)
    builder.add_edge("generate",       "citation_check")
    builder.add_edge("rewrite",        "retrieve")

    builder.add_conditional_edges(
        "filter_docs", route_after_filter,
        {"generate": "generate", "no_info": "no_info"},
    )
    builder.add_conditional_edges(
        "citation_check", route_after_citation,
        {"verify_and_evaluate": "verify_and_evaluate", "rewrite": "rewrite", "__end__": END},
    )
    builder.add_conditional_edges(
        "verify_and_evaluate", route_after_verify_and_evaluate,
        {"__end__": END, "rewrite": "rewrite"},
    )

    # ── ORIGINAL routing wiring (commented out) ───────────────────────────────
    # builder.add_conditional_edges(
    #     "citation_check", route_after_citation,
    #     {"verify_answer": "verify_answer", "rewrite": "rewrite", "__end__": END},
    # )
    # builder.add_conditional_edges(
    #     "verify_answer", route_after_verify,
    #     {"evaluate": "evaluate", "rewrite": "rewrite", "__end__": END},
    # )
    # builder.add_conditional_edges(
    #     "evaluate", route_after_evaluate,
    #     {"__end__": END, "rewrite": "rewrite"},
    # )
    # ─────────────────────────────────────────────────────────────────────────

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


def chat(
    question: str,
    session_id: str,
    extra_callbacks: Optional[list] = None,
) -> tuple[str, AgentState]:
    """Run the pipeline and return (answer, final_state).

    final_state contains _node_timings, quality_score, has_citations,
    retry_count — all needed by api.py to build RequestMetrics.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.recursion_limit,
    }
    if extra_callbacks:
        config["callbacks"] = extra_callbacks

    final = graph.invoke(
        {
            "messages":         [HumanMessage(content=question)],
            "current_question": question,
            "quality_score":    0,
            "retry_count":      0,
            "retrieved_docs":   [],
            "filtered_docs":    [],
            "answer_grounded":  False,
            "has_citations":    False,
            "_node_timings":    {},
        },
        config=config,
    )
    last_ai = next(m for m in reversed(final["messages"]) if isinstance(m, AIMessage))
    return last_ai.content, final
