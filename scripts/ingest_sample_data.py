"""
CLI Ingestion Script
Indexes enterprise documents into Dense (Qdrant) and Sparse (BM25) search indices.
"""

import os
import sys

# Ensure enterprise_rag is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enterprise_rag.core.document_loader import DocumentLoader
from enterprise_rag.core.chunking import RecursiveTokenChunker
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.config import settings

def main():
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_enterprise_docs"))
    print("=" * 70)
    print("  ENTERPRISE DOCUMENT INGESTION PIPELINE")
    print("=" * 70)
    print(f"Target Directory: {docs_dir}")

    if not os.path.exists(docs_dir):
        print(f"Error: Directory not found: {docs_dir}")
        return

    # 1. Load documents
    docs = DocumentLoader.load_directory(docs_dir, extensions=[".md", ".txt", ".pdf"])
    print(f"\n[+] Loaded {len(docs)} documents:")
    for d in docs:
        print(f"  - {d.metadata.get('file_name')} ({len(d.content)} chars)")

    # 2. Chunk documents
    chunker = RecursiveTokenChunker(chunk_size=384, chunk_overlap=48)
    chunks = chunker.split_documents(docs)
    print(f"\n[+] Generated {len(chunks)} recursive token chunks.")

    # 3. Index into Hybrid Retriever
    retriever = HybridRetriever()
    print("\n[+] Indexing chunks into Qdrant Vector Store & BM25 Sparse Index...")
    indexed = retriever.index_chunks(chunks)
    print(f"[✓] Successfully indexed {indexed} chunks across dense and sparse engines.")
    print("=" * 70)

if __name__ == "__main__":
    main()
