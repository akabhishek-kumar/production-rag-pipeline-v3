"""Langfuse tracing + LangChain token-count callback — v3.

Langfuse is an open-source LLM observability platform.
It captures every prompt, completion, and token count through the pipeline
and displays them in a real-time dashboard.

If LANGFUSE_PUBLIC_KEY is not set, tracing is silently disabled —
the pipeline still runs normally without any external dependency.

Token counting:
  TokenCountCallback is a LangChain BaseCallbackHandler that hooks into
  on_llm_end to accumulate input + output tokens across all LLM calls
  in a single graph execution. The counts are used to estimate cost.

Setup:
  1. Sign up at https://cloud.langfuse.com (free tier)
  2. Create a project → copy public/secret keys
  3. Add to .env:
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...
       LANGFUSE_HOST=https://cloud.langfuse.com
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings

if TYPE_CHECKING:
    pass

# ── Optional Langfuse import ──────────────────────────────────────────────────

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

_langfuse_client: Optional["Langfuse"] = None


def _get_langfuse() -> Optional["Langfuse"]:
    """Lazy-init Langfuse client — returns None if keys not set."""
    global _langfuse_client
    if not _LANGFUSE_AVAILABLE:
        return None
    if not settings.langfuse_public_key:
        return None
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _langfuse_client


def get_langfuse_handler(
    trace_id: str,
    session_id: str,
) -> Optional["LangfuseCallbackHandler"]:
    """Return a per-request Langfuse callback handler, or None if disabled.

    When attached to a LangChain/LangGraph run, it automatically captures:
      - Every prompt sent to the LLM
      - Every completion received
      - Token counts per call
      - Node-level latency within the graph
      - The full trace in the Langfuse dashboard
    """
    client = _get_langfuse()
    if client is None:
        return None
    return LangfuseCallbackHandler(
        trace_id=trace_id,
        session_id=session_id,
        tags=["production-rag-v3"],
        metadata={"version": "3"},
    )


def langfuse_enabled() -> bool:
    return _get_langfuse() is not None


# ── Token counting callback ───────────────────────────────────────────────────


class TokenCountCallback(BaseCallbackHandler):
    """Accumulates input + output tokens across all LLM calls in one run.

    Attach to graph.invoke() via the callbacks list:
        cb = TokenCountCallback()
        result = graph.invoke(input, config={"callbacks": [cb]})
        print(cb.input_tokens, cb.output_tokens)

    Works with any LangChain LLM that returns token_usage in llm_output.
    Groq returns:
        response.llm_output["token_usage"]["prompt_tokens"]
        response.llm_output["token_usage"]["completion_tokens"]
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not response.llm_output:
            return
        usage = response.llm_output.get("token_usage", {})
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
