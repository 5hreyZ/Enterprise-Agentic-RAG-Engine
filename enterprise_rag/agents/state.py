from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enterprise_rag.retrieval.dense import SearchResult

@dataclass
class AgentState:
    """
    Unified state machine object tracking the full lifecycle of an Agentic RAG request.
    Inspired by LangGraph state definitions for deterministic observability and debugging.
    """
    # Inputs
    user_query: str
    
    # Query Analysis & Decomposition
    is_multi_hop: bool = False
    sub_queries: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)
    
    # Retrieval State
    current_hop: int = 0
    max_hops: int = 3
    retrieved_documents: List[SearchResult] = field(default_factory=list)
    filtered_documents: List[SearchResult] = field(default_factory=list)
    
    # Intermediate Reasoning & Scratchpad
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_answers: List[str] = field(default_factory=list)
    
    # Generation & Guardrails
    final_answer: str = ""
    is_faithful: bool = True
    faithfulness_score: float = 1.0
    unsupported_claims: List[str] = field(default_factory=list)
    verification_verdict: str = "PASS"  # PASS, RE-RETRIEVE, FAIL
    
    # Execution Metadata
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None

    def add_trace(self, step_name: str, details: Dict[str, Any]):
        """Record an execution step in the agent reasoning trace."""
        self.reasoning_trace.append({
            "step": step_name,
            "hop": self.current_hop,
            "details": details,
        })
