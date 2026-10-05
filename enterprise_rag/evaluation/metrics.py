import re
from typing import List, Dict, Any

class RAGMetrics:
    """
    Evaluation metrics for enterprise RAG pipelines based on RAGAS framework principles.
    """

    @staticmethod
    def calculate_context_relevance(
        retrieved_contexts: List[str],
        ground_truth_relevant_keywords: List[str],
    ) -> float:
        """
        Computes Context Relevance / Precision:
        Ratio of retrieved sentences/passages containing essential factual keywords.
        """
        if not retrieved_contexts or not ground_truth_relevant_keywords:
            return 0.0

        hits = 0
        total = len(retrieved_contexts)
        for ctx in retrieved_contexts:
            ctx_lower = ctx.lower()
            if any(kw.lower() in ctx_lower for kw in ground_truth_relevant_keywords):
                hits += 1

        return hits / max(1, total)

    @staticmethod
    def calculate_answer_faithfulness(
        generated_answer: str,
        retrieved_contexts: List[str],
    ) -> float:
        """
        Computes Answer Faithfulness:
        Ratio of claims in the generated answer directly supported by the retrieved context.
        """
        if not generated_answer or not retrieved_contexts:
            return 0.0

        # Break answer into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", generated_answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        combined_ctx = " ".join(retrieved_contexts).lower()
        supported = 0

        for sentence in sentences:
            tokens = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
            if not tokens:
                supported += 1
                continue
            # Check token recall in context
            token_hits = sum(1 for t in tokens if t in combined_ctx)
            if token_hits / len(tokens) >= 0.5:
                supported += 1

        return supported / max(1, len(sentences))

    @staticmethod
    def calculate_summary_stats(metric_values: List[float]) -> Dict[str, float]:
        """Compute mean, min, and max for a metric series."""
        if not metric_values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": sum(metric_values) / len(metric_values),
            "min": min(metric_values),
            "max": max(metric_values),
        }
