import unittest
from enterprise_rag.core.document_loader import Document
from enterprise_rag.core.chunking import RecursiveTokenChunker
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.sparse import BM25SparseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever, ReciprocalRankFusion

class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        docs = [
            Document(content="Kubernetes cluster uses Calico CNI with strict network policies.", metadata={"topic": "k8s"}),
            Document(content="PostgreSQL database replication lag must not exceed 500 milliseconds.", metadata={"topic": "db"}),
            Document(content="Payment service requires PCI-DSS Level 1 tokenization and AES-256.", metadata={"topic": "pci"}),
        ]
        chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=10)
        self.chunks = chunker.split_documents(docs)

    def test_dense_retrieval(self):
        dense = DenseRetriever()
        dense.index_chunks(self.chunks)
        results = dense.search("PostgreSQL replication", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].source, "dense")

    def test_sparse_bm25_retrieval(self):
        bm25 = BM25SparseRetriever()
        bm25.index_chunks(self.chunks)
        results = bm25.search("PCI-DSS tokenization", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("pci", results[0].text.lower())
        self.assertEqual(results[0].source, "sparse")

    def test_reciprocal_rank_fusion(self):
        hybrid = HybridRetriever()
        hybrid.index_chunks(self.chunks)
        fused = hybrid.search("Calico network policies", top_k=2)
        self.assertGreater(len(fused), 0)
        self.assertEqual(fused[0].source, "hybrid_rrf")
        self.assertTrue(fused[0].score > 0.0)

if __name__ == "__main__":
    unittest.main()
