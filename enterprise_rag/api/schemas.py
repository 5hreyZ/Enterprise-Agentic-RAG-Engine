from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# If pydantic is available, inherit from BaseModel, otherwise use dataclass
try:
    from pydantic import BaseModel, Field
    
    class QueryRequest(BaseModel):
        query: str = Field(..., description="Enterprise query string")
        stream: bool = Field(default=False, description="Stream answer tokens via SSE")
        max_hops: int = Field(default=3, description="Maximum reasoning hops")

    class IngestRequest(BaseModel):
        documents: List[Dict[str, Any]] = Field(..., description="List of raw document payloads")
        chunk_size: int = Field(default=512, description="Target chunk size in tokens")
        chunk_overlap: int = Field(default=64, description="Chunk overlap tokens")

    class QueryResponse(BaseModel):
        query: str
        answer: str
        is_multi_hop: bool
        sub_queries: List[str]
        citations: List[Dict[str, Any]]
        faithfulness_score: float
        verification_verdict: str
        latency_ms: float
        reasoning_trace: List[Dict[str, Any]]

    class HealthResponse(BaseModel):
        status: str
        version: str
        qdrant_connected: bool
        vllm_connected: bool

except ImportError:
    @dataclass
    class QueryRequest:
        query: str
        stream: bool = False
        max_hops: int = 3

    @dataclass
    class IngestRequest:
        documents: List[Dict[str, Any]]
        chunk_size: int = 512
        chunk_overlap: int = 64

    @dataclass
    class QueryResponse:
        query: str
        answer: str
        is_multi_hop: bool
        sub_queries: List[str]
        citations: List[Dict[str, Any]]
        faithfulness_score: float
        verification_verdict: str
        latency_ms: float
        reasoning_trace: List[Dict[str, Any]]

    @dataclass
    class HealthResponse:
        status: str
        version: str
        qdrant_connected: bool
        vllm_connected: bool
