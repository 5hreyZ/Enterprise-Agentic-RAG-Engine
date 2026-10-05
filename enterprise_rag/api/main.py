"""
FastAPI Enterprise Application Entrypoint
Provides high-performance async REST and SSE streaming endpoints for Agentic RAG.
"""

from enterprise_rag.config import settings
from enterprise_rag.api.routes import handle_query, handle_stream_query, handle_ingest, service
from enterprise_rag.evaluation.evaluator import BenchmarkEvaluator

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from enterprise_rag.api.schemas import QueryRequest, IngestRequest, QueryResponse, HealthResponse

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description="Autonomous Agentic RAG Engine with Llama-3-8B, Hybrid Search (BM25 + ColBERT), and vLLM Serving.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            qdrant_connected=service.retriever.dense._qdrant_client is not None,
            vllm_connected=settings.LLM_PROVIDER == "vllm",
        )

    @app.get("/metrics")
    async def metrics():
        return {
            "retriever": {
                "sparse_docs_indexed": service.retriever.sparse.num_docs,
                "vector_dimension": service.retriever.dense.embedding_engine.dimension,
            },
            "system": {
                "environment": settings.ENVIRONMENT,
                "hybrid_rrf_k": settings.HYBRID_RRF_K,
                "relevance_threshold": settings.RELEVANCE_THRESHOLD,
                "faithfulness_threshold": settings.FAITHFULNESS_THRESHOLD,
            }
        }

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest):
        try:
            result = handle_query(req.query, max_hops=req.max_hops)
            return QueryResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/query/stream")
    async def query_stream_endpoint(req: QueryRequest):
        return StreamingResponse(
            handle_stream_query(req.query),
            media_type="text/event-stream",
        )

    @app.post("/api/v1/documents/ingest")
    async def ingest_endpoint(req: IngestRequest):
        try:
            result = handle_ingest(req.documents, req.chunk_size, req.chunk_overlap)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/evaluate")
    async def evaluate_endpoint():
        # Built-in synthetic enterprise multi-hop evaluation set
        eval_cases = [
            {
                "query": "What are the SLA penalty thresholds for cloud compute and what is the required incident response time?",
                "ground_truth_keywords": ["sla", "uptime", "penalty", "tier", "incident", "response"],
                "is_multi_hop": True,
            },
            {
                "query": "Compare SOC2 Type II data retention requirements with the quarterly financial audit compliance guidelines.",
                "ground_truth_keywords": ["soc2", "retention", "audit", "compliance", "policy"],
                "is_multi_hop": True,
            }
        ]
        evaluator = BenchmarkEvaluator(service.orchestrator)
        report = evaluator.run_benchmark(eval_cases)
        return report

except ImportError:
    # Minimal WSGI/ASGI stub for environments without fastapi installed
    app = None

def get_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enterprise_rag.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
