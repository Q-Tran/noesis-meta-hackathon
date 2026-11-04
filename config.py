"""Central configuration for Noesis, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ModuleNotFoundError:
    pass  # python-dotenv optional; env vars still read from the environment


def _get(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Graph Database (Neo4j)
    neo4j_uri: str = _get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _get("NEO4J_USER", "neo4j")
    neo4j_password: str = _get("NEO4J_PASSWORD", "password")

    # Vector Storage (Milvus)
    milvus_uri: str = _get("MILVUS_URI", "http://localhost:19530")
    # Milvus is an accelerator, never a gate: every semantic path degrades to
    # keyword-only without it. Keep the connect bounded so an unreachable
    # server fails fast instead of stalling the graph write that triggered a
    # reindex (pymilvus otherwise blocks ~10s per attempt).
    milvus_timeout: float = float(_get("MILVUS_TIMEOUT_SECONDS", "2.0"))
    
    # Ingestion (Kafka)
    kafka_brokers: str = _get("KAFKA_BROKERS", "localhost:9092")

    # Tool execution spans + trace outcomes (written to the graph)
    enable_skill_tracing: bool = _get_bool("ENABLE_SKILL_TRACING", True)

    # LLM
    llm_provider: str = _get("LLM_PROVIDER", "null")  # null | openai | ollama | litellm | att
    litellm_model: str = _get("LITELLM_MODEL", "gpt-4o-mini") # Can be "ollama/llama3.1" etc.
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = _get("OPENAI_BASE_URL", "")
    llm_timeout: float = float(_get("LLM_TIMEOUT", "600"))
    ollama_host: str = _get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1")

    # AT&T Provider (via local proxy)
    att_proxy_url: str = _get("ATT_PROXY_URL", "http://127.0.0.1:8000/v1")
    att_model: str = _get("ATT_MODEL", "gpt-5.4")

    # Embeddings
    embedding_model: str = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_model_path: str = _get("EMBEDDING_MODEL_PATH", "models/all-MiniLM-L6-v2")
    embedding_dimension: int = int(_get("EMBEDDING_DIMENSION", "384"))

    # Reranking
    enable_reranking: bool = _get_bool("ENABLE_RERANKING", True)
    reranker_model: str = _get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_model_path: str = _get(
        "RERANKER_MODEL_PATH", "models/ms-marco-MiniLM-L-6-v2"
    )
    rerank_candidate_multiplier: int = int(_get("RERANK_CANDIDATE_MULTIPLIER", "10"))
    rerank_min_candidates: int = int(_get("RERANK_MIN_CANDIDATES", "50"))
    rerank_max_candidates: int = int(_get("RERANK_MAX_CANDIDATES", "100"))
    dedupe_rerank_candidates: int = int(_get("DEDUPE_RERANK_CANDIDATES", "25"))

    # Entity Resolution & Argumentation
    resolution_top_k: int = int(_get("RESOLUTION_TOP_K", "3"))


settings = Settings()
