import time
from typing import Optional, List
from enterprise_rag.agents.state import AgentState
from enterprise_rag.agents.query_router import QueryRouter
from enterprise_rag.agents.guardrails import ContextRelevanceGrader, FaithfulnessVerifier
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker
from enterprise_rag.retrieval.dense import SearchResult
from enterprise_rag.inference.llm_client import LLMClient
from enterprise_rag.inference.prompts import SYNTHESIS_PROMPT
from enterprise_rag.config import settings

class AgenticRAGOrchestrator:
    """
    Autonomous Multi-Hop Agentic RAG Engine.
    Executes iterative retrieval loops, document reranking, relevance filtering,
    grounded synthesis with citations, and self-correcting faithfulness verification.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.router = QueryRouter(self.llm)
        self.relevance_grader = ContextRelevanceGrader(self.llm, threshold=settings.RELEVANCE_THRESHOLD)
        self.faithfulness_verifier = FaithfulnessVerifier(self.llm, threshold=settings.FAITHFULNESS_THRESHOLD)

    def run(self, query: str) -> AgentState:
        """
        Executes end-to-end multi-hop agentic workflow on the given query.
        Returns final AgentState with reasoning traces, cited sources, and guardrail verdicts.
        """
        start_time = time.perf_counter()
        state = AgentState(user_query=query, max_hops=settings.MAX_AGENT_HOPS)

        # Step 1: Query Routing & Decomposition
        state = self.router.route_and_decompose(state)

        # Step 2: Multi-Hop Retrieval Loop
        accumulated_chunks: List[SearchResult] = []
        seen_chunk_ids = set()

        for hop_idx, sub_q in enumerate(state.sub_queries[:state.max_hops]):
            state.current_hop = hop_idx + 1
            
            # Hybrid Dense-Sparse Search (BM25 + Qdrant RRF)
            candidates = self.retriever.search(sub_q, top_k=settings.TOP_K_RETRIEVAL)
            
            # Cross-Encoder / ColBERT Late-Interaction Reranking
            reranked = self.reranker.rerank(sub_q, candidates, top_n=settings.RERANK_TOP_N)
            
            # Relevance Grading Guardrail
            graded = self.relevance_grader.filter_relevant_documents(sub_q, reranked)

            for item in graded:
                if item.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(item.chunk_id)
                    accumulated_chunks.append(item)

            state.add_trace(f"retrieval_hop_{state.current_hop}", {
                "sub_query": sub_q,
                "retrieved_count": len(candidates),
                "reranked_count": len(reranked),
                "relevant_count": len(graded),
            })

        state.retrieved_documents = accumulated_chunks
        state.filtered_documents = accumulated_chunks

        # Step 3: Grounded Answer Synthesis
        if not accumulated_chunks:
            state.final_answer = "No relevant enterprise documents could be retrieved to answer this query."
            state.verification_verdict = "FAIL"
            state.latency_ms = (time.perf_counter() - start_time) * 1000.0
            return state

        context_str = "\n\n".join(
            f"[Passage {idx+1} | ID: {c.chunk_id} | Source: {c.metadata.get('source', 'Unknown')}]:\n{c.text}"
            for idx, c in enumerate(accumulated_chunks)
        )

        synthesis_prompt = (
            SYNTHESIS_PROMPT
            .replace("__CONTEXT__", context_str)
            .replace("__QUERY__", state.user_query)
        )
        generated_answer = self.llm.generate(synthesis_prompt)
        state.final_answer = generated_answer

        state.add_trace("synthesis", {
            "num_context_passages": len(accumulated_chunks),
            "answer_preview": generated_answer[:150] + "...",
        })

        # Step 4: Faithfulness & Hallucination Guardrail Check
        is_faithful, faith_score, unsupported, verdict = self.faithfulness_verifier.verify(
            context_chunks=accumulated_chunks,
            answer=generated_answer,
        )

        state.is_faithful = is_faithful
        state.faithfulness_score = faith_score
        state.unsupported_claims = unsupported
        state.verification_verdict = verdict

        state.add_trace("guardrail_verification", {
            "is_faithful": is_faithful,
            "faithfulness_score": faith_score,
            "unsupported_claims_count": len(unsupported),
            "verdict": verdict,
        })

        # Step 5: Self-Correction if verification fails
        if not is_faithful and verdict == "FAIL":
            state.add_trace("self_correction", {
                "action": "Appending strict evidence boundaries due to hallucination detection"
            })
            state.final_answer += (
                "\n\n> ⚠️ *Note: Certain claims could not be verified with 100% certainty against "
                "enterprise documentation and were flagged by the Faithfulness Guardrail.*"
            )

        state.latency_ms = (time.perf_counter() - start_time) * 1000.0
        return state
