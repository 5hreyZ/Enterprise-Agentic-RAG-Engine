import unittest
from enterprise_rag.core.document_loader import Document
from enterprise_rag.core.chunking import RecursiveTokenChunker

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.chunker = RecursiveTokenChunker(chunk_size=50, chunk_overlap=10)

    def test_document_creation(self):
        doc = Document(content="Sample enterprise policy text.", metadata={"source": "policy.md"})
        self.assertEqual(doc.content, "Sample enterprise policy text.")
        self.assertEqual(doc.metadata["source"], "policy.md")
        self.assertTrue(len(doc.doc_id) > 0)

    def test_recursive_split_short_text(self):
        doc = Document(content="Short document under token limit.", metadata={"source": "test.txt"})
        chunks = self.chunker.split_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short document under token limit.")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].doc_id, doc.doc_id)

    def test_recursive_split_long_text(self):
        long_text = "Section 1: Security Requirements. " * 30
        doc = Document(content=long_text, metadata={"source": "sec.md"})
        chunks = self.chunker.split_document(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c.chunk_id)
            self.assertEqual(c.doc_id, doc.doc_id)
            self.assertIn("chunk_index", c.metadata)

if __name__ == "__main__":
    unittest.main()
