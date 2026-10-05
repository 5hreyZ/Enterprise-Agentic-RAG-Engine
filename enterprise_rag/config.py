import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Settings:
    """Enterprise RAG Engine configuration parameters."""
    
    # Application & Environment
    PROJECT_NAME: str = "Enterprise Agentic RAG Engine"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    API_PORT: int = int(os.getenv("PORT", "8000"))
    API_HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # Vector Database (Qdrant)
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "enterprise_knowledge_base")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))  # 384 for all-MiniLM-L6-v2, 1024 for BGE-M3
    
    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # cuda, cpu, mps
    
    # Sparse Retrieval (BM25)
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))
    
    # Hybrid Retrieval & Fusion
    HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))  # Constant k for Reciprocal Rank Fusion
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "4"))
    
    # Reranker Configuration
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Inference Engine (vLLM / Hugging Face / OpenAI-compatible endpoint)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "vllm")  # vllm, hf, mock
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct-AWQ")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    
    # Multi-hop Agent Parameters
    MAX_AGENT_HOPS: int = int(os.getenv("MAX_AGENT_HOPS", "3"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.65"))
    FAITHFULNESS_THRESHOLD: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.85"))

settings = Settings()
