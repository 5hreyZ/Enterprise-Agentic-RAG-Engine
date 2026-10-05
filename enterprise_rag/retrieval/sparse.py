import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional
from enterprise_rag.core.chunking import TextChunk
from enterprise_rag.retrieval.dense import SearchResult
from enterprise_rag.config import settings

class BM25SparseRetriever:
    """
    BM25Okapi sparse lexical retriever.
    Captures exact keyword matches, SKU numbers, code tokens, and acronyms
    that dense vector embeddings often compress or dilute.
    """

    def __init__(self, k1: Optional[float] = None, b: Optional[float] = None):
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        self.corpus_chunks: List[TextChunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.num_docs: int = 0
        self.tokenized_corpus: List[List[str]] = []
        self._rank_bm25_model = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust regex tokenizer for alphanumeric words and code symbols."""
        return re.findall(r"\b\w+\b", text.lower())

    def index_chunks(self, chunks: List[TextChunk]) -> int:
        """Build BM25 index over the provided text chunks."""
        self.corpus_chunks = chunks
        self.num_docs = len(chunks)
        if self.num_docs == 0:
            return 0

        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.num_docs)

        # Build document frequencies
        self.doc_freqs = Counter()
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        # Optionally use rank_bm25 if available
        try:
            from rank_bm25 import BM25Okapi
            self._rank_bm25_model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        except ImportError:
            self._rank_bm25_model = None

        return self.num_docs

    def _compute_idf(self, term: str) -> float:
        """Compute Lucene-style BM25 IDF for a term."""
        df = self.doc_freqs.get(term, 0)
        # BM25Okapi smooth IDF formula
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """Compute Okapi BM25 score for a specific document against query terms."""
        doc = self.tokenized_corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        term_counts = Counter(doc)
        score = 0.0

        for term in query_terms:
            if term not in term_counts:
                continue
            tf = term_counts[term]
            idf = self._compute_idf(term)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search BM25 index for given query."""
        if self.num_docs == 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        if self._rank_bm25_model is not None:
            scores = self._rank_bm25_model.get_scores(query_terms)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for idx in ranked_indices:
                score = float(scores[idx])
                if score <= 0.0:
                    continue
                c = self.corpus_chunks[idx]
                results.append(
                    SearchResult(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        text=c.text,
                        score=score,
                        metadata=c.metadata,
                        source="sparse",
                    )
                )
            return results

        # Native BM25 scoring
        scored = []
        for i in range(self.num_docs):
            score = self._score_document(query_terms, i)
            if score > 0.0:
                scored.append((score, self.corpus_chunks[i]))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_hits = scored[:top_k]

        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(score),
                metadata=chunk.metadata,
                source="sparse",
            )
            for score, chunk in top_hits
        ]
