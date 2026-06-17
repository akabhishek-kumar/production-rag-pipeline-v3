"""Entry point — production RAG pipeline v2.

IMPORTANT: Run `python ingest.py` before starting the server.
The server will fail with FileNotFoundError if chroma_db / bm25_index.pkl do not exist.

Start: uvicorn main:app --reload
Docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from app.api import router
from app.config import settings

app = FastAPI(
    title="Production RAG Pipeline v2",
    description=(
        "Production-grade RAG with hybrid retrieval (BM25 + vector), "
        "cross-encoder reranking, citation enforcement, hallucination verification, "
        "and quality-gated retries."
    ),
    version="2.0.0",
)

app.include_router(router)


@app.get("/health", tags=["meta"])
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "model": settings.groq_model,
        "reranker": settings.reranker_model,
        "persist_dir": settings.persist_dir,
    }
