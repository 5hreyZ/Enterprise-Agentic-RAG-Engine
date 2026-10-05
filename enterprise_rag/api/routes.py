import json
from typing import Dict, Any, Generator
from enterprise_rag.agents.multi_hop import AgenticRAGOrchestrator
from enterprise_rag.core.document_loader import Document
from enterprise_rag.core.chunking import RecursiveTokenChunker
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker
from enterprise_rag.inference.llm_client import LLMClient
from enterprise_rag.config import settings

class RAGService:
    """Singleton service manager holding initialized orchestrator and retrievers."""
    _instance = None

    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.llm = LLMClient()
        self.orchestrator = AgenticRAGOrchestrator(
            hybrid_retriever=self.retriever,
            reranker=self.reranker,
            llm_client=self.llm,
        )
        self.chunker = RecursiveTokenChunker()

    @classmethod
    def get_instance(cls) -> "RAGService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

service = RAGService.get_instance()

def handle_query(query: str, max_hops: int = 3) -> Dict[str, Any]:
    """Execute multi-hop agent query and format structured API response."""
    state = service.orchestrator.run(query)
    
    citations = [
        {
            "chunk_id": doc.chunk_id,
            "doc_id": doc.doc_id,
            "source": doc.metadata.get("source", "Unknown"),
            "score": round(doc.score, 4),
            "text_preview": doc.text[:180] + "...",
        }
        for doc in state.filtered_documents
    ]

    return {
        "query": state.user_query,
        "answer": state.final_answer,
        "is_multi_hop": state.is_multi_hop,
        "sub_queries": state.sub_queries,
        "citations": citations,
        "faithfulness_score": round(state.faithfulness_score, 3),
        "verification_verdict": state.verification_verdict,
        "latency_ms": round(state.latency_ms, 2),
        "reasoning_trace": state.reasoning_trace,
    }

def handle_stream_query(query: str) -> Generator[str, None, None]:
    """Stream token chunks via Server-Sent Events (SSE) format."""
    state = service.orchestrator.run(query)
    # Stream the answer tokens
    words = state.final_answer.split(" ")
    for word in words:
        payload = json.dumps({"token": word + " "})
        yield f"data: {payload}\n\n"
    
    # Send final metadata event
    meta_payload = json.dumps({
        "event": "complete",
        "faithfulness": state.faithfulness_score,
        "verdict": state.verification_verdict,
        "latency_ms": state.latency_ms,
    })
    yield f"data: {meta_payload}\n\n"

def handle_ingest(documents_data: list, chunk_size: int = 512, chunk_overlap: int = 64) -> Dict[str, Any]:
    """Ingest, chunk, and index enterprise documents into Qdrant & BM25."""
    docs = [
        Document(
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
            doc_id=d.get("doc_id", ""),
        )
        for d in documents_data
    ]
    chunker = RecursiveTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_documents(docs)
    indexed_count = service.retriever.index_chunks(chunks)

    return {
        "status": "success",
        "documents_processed": len(docs),
        "chunks_created": len(chunks),
        "chunks_indexed": indexed_count,
    }
