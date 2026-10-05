import unittest
from enterprise_rag.core.document_loader import Document
from enterprise_rag.core.chunking import RecursiveTokenChunker
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker
from enterprise_rag.inference.llm_client import LLMClient
from enterprise_rag.agents.multi_hop import AgenticRAGOrchestrator
from enterprise_rag.agents.guardrails import ContextRelevanceGrader, FaithfulnessVerifier
from enterprise_rag.retrieval.dense import SearchResult

class TestAgenticFlow(unittest.TestCase):
    def setUp(self):
        docs = [
            Document(content="SOC2 Type II requires 7 years of immutable log retention.", metadata={"source": "soc2.md"}),
            Document(content="Sev-1 incidents require on-call response within 15 minutes.", metadata={"source": "sla.md"}),
        ]
        chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_documents(docs)

        self.retriever = HybridRetriever()
        self.retriever.index_chunks(chunks)
        self.reranker = CrossEncoderReranker()
        self.llm = LLMClient()
        self.orchestrator = AgenticRAGOrchestrator(
            hybrid_retriever=self.retriever,
            reranker=self.reranker,
            llm_client=self.llm,
        )

    def test_query_routing_and_decomposition(self):
        query = "What is the Sev-1 response time and how long must SOC2 logs be retained?"
        state = self.orchestrator.run(query)

        self.assertTrue(state.is_multi_hop)
        self.assertGreaterEqual(len(state.sub_queries), 2)
        self.assertGreater(len(state.retrieved_documents), 0)
        self.assertTrue(state.final_answer)
        self.assertTrue(state.is_faithful)
        self.assertEqual(state.verification_verdict, "PASS")
        self.assertGreater(state.faithfulness_score, 0.8)
        self.assertGreater(len(state.reasoning_trace), 3)

    def test_relevance_grader(self):
        grader = ContextRelevanceGrader(self.llm)
        chunk = SearchResult(chunk_id="c1", doc_id="d1", text="Sev-1 incidents require 15 minutes response.", score=0.8)
        is_rel, conf, _ = grader.grade_document("Sev-1 response time", chunk)
        self.assertTrue(is_rel)
        self.assertGreaterEqual(conf, 0.6)

    def test_faithfulness_verifier(self):
        verifier = FaithfulnessVerifier(self.llm)
        chunks = [SearchResult(chunk_id="c1", doc_id="d1", text="Data retention is 7 years in Glacier.", score=0.9)]
        is_faith, score, unsupported, verdict = verifier.verify(chunks, "The enterprise data retention policy mandates 7 years.")
        self.assertTrue(is_faith)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(len(unsupported), 0)

if __name__ == "__main__":
    unittest.main()
