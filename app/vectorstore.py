"""Hybrid retriever + cross-encoder reranker — v2.

v2 changes vs v1:
  - Hybrid retrieval: BM25 (keyword) + Chroma (semantic) merged via Reciprocal
    Rank Fusion (RRF). BM25 catches exact keyword matches that vector search misses
    (acronyms, proper nouns, rare terms). RRF merges the ranked lists without
    needing EnsembleRetriever (deprecated in LangChain 0.3+).
  - Cross-encoder reranking: CrossEncoder scores (query, doc) pairs jointly,
    giving much better relevance than cosine similarity which encodes separately.
  - Lazy init: vectorstore and reranker load on first request, not at import time,
    so tests can patch before the real models are loaded.

Startup flow:
  ingest.py  → embed docs → save chroma_db/ + bm25_index.pkl  (run once)
  app starts → load_vectorstore() + load_bm25_documents()       (instant)
  first query → get_reranker() downloads cross-encoder ~80MB    (once, cached)
"""

import os
import pickle
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder

from app.config import settings

_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_reranker: Optional[CrossEncoder] = None


def _rrf_merge(*doc_lists: list[Document], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion — merges ranked lists from multiple retrievers.

    RRF score for a document = sum of 1/(k + rank) across all lists it appears in.
    k=60 is the standard constant from the original RRF paper (Cormack et al. 2009).
    Documents are deduplicated by content; the highest-scored copy is kept.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content[:200]          # content fingerprint
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=_EMBEDDING_MODEL)


def get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder (downloads ~80MB on first call, cached after)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def load_vectorstore() -> Chroma:
    """Load persisted Chroma vector store from disk."""
    if not os.path.exists(settings.persist_dir):
        raise FileNotFoundError(
            f"Vector store not found at '{settings.persist_dir}'. "
            "Run `python ingest.py` first."
        )
    return Chroma(
        persist_directory=settings.persist_dir,
        embedding_function=get_embeddings(),
        collection_name="production_rag_v2",
    )


def load_bm25_documents() -> list[Document]:
    """Load serialized chunks for BM25 (written by ingest.py)."""
    if not os.path.exists(settings.bm25_index_path):
        raise FileNotFoundError(
            f"BM25 index not found at '{settings.bm25_index_path}'. "
            "Run `python ingest.py` first."
        )
    with open(settings.bm25_index_path, "rb") as f:
        return pickle.load(f)


def get_hybrid_retriever(
    vectorstore: Chroma,
    documents: list[Document],
) -> RunnableLambda:
    """Combine BM25 (keyword) + Chroma (semantic) via Reciprocal Rank Fusion.

    Both retrievers run independently, then RRF merges their ranked lists.
    BM25 catches exact keyword matches (acronyms, proper nouns, version numbers)
    that vector search misses. Vector search catches semantic meaning BM25 misses.
    RRF is used instead of EnsembleRetriever to avoid langchain.retrievers
    deprecation in LangChain 0.3+.
    """
    bm25_retriever = BM25Retriever.from_documents(documents, k=settings.bm25_k)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.vector_k},
    )

    def hybrid_search(query: str) -> list[Document]:
        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vector_retriever.invoke(query)
        return _rrf_merge(bm25_docs, vector_docs)

    return RunnableLambda(hybrid_search)


def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    """Cross-encoder reranking — better precision than cosine similarity.

    The cross-encoder sees the (query, document) pair together in one forward
    pass, letting it model query-document interactions that bi-encoders miss.
    Scores are attached to metadata for observability in logs and evaluations.
    """
    if not docs:
        return docs

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_k = ranked[: settings.reranker_k]

    result = []
    for doc, score in top_k:
        doc.metadata["reranker_score"] = round(float(score), 4)
        result.append(doc)

    return result
