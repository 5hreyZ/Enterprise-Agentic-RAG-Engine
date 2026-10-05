from typing import List, Dict, Optional
from enterprise_rag.core.chunking import TextChunk
from enterprise_rag.retrieval.dense import DenseRetriever, SearchResult
from enterprise_rag.retrieval.sparse import BM25SparseRetriever
from enterprise_rag.config import settings

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine rankings from multiple retrievers:
    RRF(d) = sum_{m in M} (1 / (k + rank_m(d)))
    """

    @staticmethod
    def fuse(
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
        top_n: int = 10,
    ) -> List[SearchResult]:
        """
        Merge and rerank results from dense and sparse retrievers using RRF.
        Returns top_n deduplicated SearchResult objects with normalized RRF scores.
        """
        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, SearchResult] = {}

        # Process dense rankings
        for rank, res in enumerate(dense_results, start=1):
            chunk_id = res.chunk_id
            item_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # Process sparse rankings
        for rank, res in enumerate(sparse_results, start=1):
            chunk_id = res.chunk_id
            if chunk_id not in item_map:
                item_map[chunk_id] = res
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # Sort by composite RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        fused_results: List[SearchResult] = []
        for chunk_id, score in sorted_chunks:
            base = item_map[chunk_id]
            fused_results.append(
                SearchResult(
                    chunk_id=base.chunk_id,
                    doc_id=base.doc_id,
                    text=base.text,
                    score=float(score),
                    metadata=base.metadata,
                    source="hybrid_rrf",
                )
            )

        return fused_results

class HybridRetriever:
    """
    Unified Hybrid Retriever combining Dense (Qdrant) and Sparse (BM25)
    with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25SparseRetriever] = None,
        rrf_k: Optional[int] = None,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or BM25SparseRetriever()
        self.rrf_k = rrf_k if rrf_k is not None else settings.HYBRID_RRF_K

    def index_chunks(self, chunks: List[TextChunk]) -> int:
        """Index chunks across both dense and sparse indices."""
        count_dense = self.dense.index_chunks(chunks)
        count_sparse = self.sparse.index_chunks(chunks)
        return max(count_dense, count_sparse)

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[SearchResult]:
        """
        Execute parallel dense and sparse searches, then merge via Reciprocal Rank Fusion.
        """
        dense_hits = self.dense.search(query, top_k=top_k * 2)
        sparse_hits = self.sparse.search(query, top_k=top_k * 2)

        return ReciprocalRankFusion.fuse(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            k=self.rrf_k,
            top_n=top_k,
        )
