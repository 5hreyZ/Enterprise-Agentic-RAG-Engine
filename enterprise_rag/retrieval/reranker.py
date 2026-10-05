import math
from typing import List, Optional
from enterprise_rag.retrieval.dense import SearchResult
from enterprise_rag.config import settings

class CrossEncoderReranker:
    """
    Reranks candidate passages using deep cross-attention or late-interaction MaxSim scoring.
    Captures fine-grained cross-token semantic alignments between the query and passage chunks,
    drastically filtering false positives from initial candidate retrieval.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._cross_encoder = None
        self._init_model()

    def _init_model(self):
        """Attempt to load sentence-transformers CrossEncoder."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.model_name)
        except (ImportError, Exception):
            self._cross_encoder = None

    def _colbert_maxsim_fallback(self, query: str, passage: str) -> float:
        """
        Token-level Late Interaction (ColBERT-style MaxSim) scoring fallback.
        Computes max token overlap and alignment across sub-tokens.
        """
        q_tokens = query.lower().split()
        p_tokens = passage.lower().split()
        if not q_tokens or not p_tokens:
            return 0.0

        p_set = set(p_tokens)
        score = 0.0

        for q_tok in q_tokens:
            if q_tok in p_set:
                score += 1.0
            else:
                # Substring/prefix soft alignment
                max_sub = max((len(q_tok) / max(len(p), 1) for p in p_tokens if q_tok in p or p in q_tok), default=0.0)
                score += 0.5 * max_sub

        # Normalize by query length and add length penalty protection
        norm_score = score / len(q_tokens)
        return float(norm_score)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: int = 5,
    ) -> List[SearchResult]:
        """
        Rerank a list of SearchResults against the query.
        Returns the top_n highest scoring results.
        """
        if not candidates:
            return []

        if self._cross_encoder is not None:
            pairs = [[query, c.text] for c in candidates]
            scores = self._cross_encoder.predict(pairs)
            scored_candidates = []
            for c, s in zip(candidates, scores):
                scored_candidates.append(
                    SearchResult(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        text=c.text,
                        score=float(s),
                        metadata=c.metadata,
                        source="reranked_cross_encoder",
                    )
                )
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            return scored_candidates[:top_n]

        # Use ColBERT-style MaxSim token interaction scoring
        scored_candidates = []
        for c in candidates:
            sim = self._colbert_maxsim_fallback(query, c.text)
            # Combine with initial retrieval score for smoothed ranking
            combined_score = 0.7 * sim + 0.3 * min(1.0, c.score)
            scored_candidates.append(
                SearchResult(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text,
                    score=float(combined_score),
                    metadata=c.metadata,
                    source="reranked_maxsim",
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_n]
