from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    groq_api_key: str = "gsk_..."
    groq_model: str = "llama-3.1-8b-instant"

    # Storage
    persist_dir: str = "./chroma_db"
    docs_dir: str = "./docs"
    bm25_index_path: str = "./bm25_index.pkl"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval — v2
    bm25_k: int = 10          # BM25 candidate pool
    vector_k: int = 10        # vector search candidate pool
    reranker_k: int = 5       # kept after cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Quality gates
    grade_threshold: int = 6
    max_retries: int = 2
    recursion_limit: int = 25

    # ── Observability — v3 ────────────────────────────────────────────────────

    # SQLite metrics database
    metrics_db_path: str = "./metrics.db"

    # Langfuse LLM tracing (optional — leave blank to disable)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # LangSmith tracing (optional — leave blank to disable)
    # Native LangGraph integration: no code changes needed, just set these vars.
    langchain_tracing_v2: str = ""   # set to "true" to enable
    langchain_api_key: str = ""      # ls__...
    langchain_project: str = "production-rag-v3"

    # CI regression gate thresholds
    ci_max_p95_latency_ms: float = 8000.0   # p95 must be under 8s
    ci_min_avg_quality: float = 5.0          # avg quality must be ≥ 5/10
    ci_min_citation_rate: float = 0.5        # ≥ 50% of answers must cite sources


settings = Settings()
