"""LLM observability integrations — v3.

Two tracing backends, both optional, both safe to disable:

┌─────────────┬────────────────────────────────────────────────────────────┐
│ Langfuse    │ Open-source. Requires a callback handler per request.      │
│             │ Self-hostable. Set LANGFUSE_PUBLIC_KEY to enable.          │
├─────────────┼────────────────────────────────────────────────────────────┤
│ LangSmith   │ Made by the LangChain team. Native LangGraph support —     │
│             │ zero code changes, just set LANGCHAIN_TRACING_V2=true and  │
│             │ LANGCHAIN_API_KEY. Auto-traces every node and LLM call.    │
└─────────────┴────────────────────────────────────────────────────────────┘

Token counting:
  TokenCountCallback is a LangChain BaseCallbackHandler that hooks into
  on_llm_end to accumulate input + output tokens across all LLM calls
  in a single graph execution. Used to estimate per-request cost.

Setup — Langfuse:
  1. Sign up at https://cloud.langfuse.com (free tier)
  2. Create a project → copy public/secret keys
  3. Add to .env:
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...

Setup — LangSmith:
  1. Sign up at https://smith.langchain.com (free tier)
  2. Settings → API Keys → Create key
  3. Add to .env:
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_API_KEY=ls__...
       LANGCHAIN_PROJECT=production-rag-v3
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings

if TYPE_CHECKING:
    pass

# ── LangSmith — activate via env vars (LangGraph reads these automatically) ──

def _activate_langsmith() -> bool:
    """Set the env vars LangChain/LangGraph check at import time.

    Returns True if LangSmith is enabled.
    LangGraph will auto-trace every node and LLM call once these are set.
    """
    if not settings.langchain_api_key or settings.langchain_tracing_v2 != "true":
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
    return True


LANGSMITH_ENABLED: bool = _activate_langsmith()

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
