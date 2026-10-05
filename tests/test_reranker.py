import unittest
from enterprise_rag.retrieval.dense import SearchResult
from enterprise_rag.retrieval.reranker import CrossEncoderReranker

class TestReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = CrossEncoderReranker()
        self.candidates = [
            SearchResult(
                chunk_id="chunk_1",
                doc_id="doc_1",
                text="The weather forecast calls for sunny skies across the region.",
                score=0.4,
            ),
            SearchResult(
                chunk_id="chunk_2",
                doc_id="doc_2",
                text="Enterprise encryption requires AES-256 for all persistent storage volumes.",
                score=0.5,
            ),
            SearchResult(
                chunk_id="chunk_3",
                doc_id="doc_3",
                text="FIPS 140-3 validation governs cryptographic key lifecycle management.",
                score=0.3,
            ),
        ]

    def test_reranker_reordering(self):
        query = "What cipher is required for persistent enterprise storage encryption?"
        reranked = self.reranker.rerank(query, self.candidates, top_n=2)
        
        self.assertEqual(len(reranked), 2)
        # chunk_2 should be ranked #1 because it mentions AES-256 and persistent storage
        self.assertEqual(reranked[0].chunk_id, "chunk_2")
        self.assertIn("reranked", reranked[0].source)
        self.assertGreater(reranked[0].score, reranked[1].score)

if __name__ == "__main__":
    unittest.main()
