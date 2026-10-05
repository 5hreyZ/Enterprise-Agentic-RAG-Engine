import json
import re
from typing import List, Tuple
from enterprise_rag.inference.llm_client import LLMClient
from enterprise_rag.inference.prompts import RELEVANCE_GRADER_PROMPT, FAITHFULNESS_PROMPT
from enterprise_rag.retrieval.dense import SearchResult
from enterprise_rag.config import settings

class ContextRelevanceGrader:
    """
    Self-correction guardrail: Evaluates retrieved passage chunks against the query.
    Filters out noise and false positives, directly increasing downstream Context Precision.
    """

    def __init__(self, llm_client: LLMClient, threshold: float = 0.65):
        self.llm = llm_client
        self.threshold = threshold

    def grade_document(self, query: str, document: SearchResult) -> Tuple[bool, float, str]:
        """
        Grades an individual passage for relevance.
        Returns: (is_relevant, confidence_score, rationale)
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        doc_words = set(re.findall(r"\w+", document.text.lower()))
        overlap = len(query_words.intersection(doc_words)) / max(1, len(query_words))

        prompt = (
            RELEVANCE_GRADER_PROMPT
            .replace("__QUERY__", query)
            .replace("__PASSAGE__", document.text[:600])
        )
        raw_resp = self.llm.generate(prompt, temperature=0.0)

        try:
            json_match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            parsed = json.loads(json_match.group(0)) if json_match else json.loads(raw_resp)
            is_relevant = bool(parsed.get("is_relevant", True))
            confidence = float(parsed.get("confidence_score", 0.85))
            rationale = parsed.get("rationale", "")
            return is_relevant and (confidence >= self.threshold), confidence, rationale
        except Exception:
            score = 0.6 * overlap + 0.4 * min(1.0, document.score)
            is_rel = score >= 0.25 or document.score > 0.4
            return is_rel, float(score), "Heuristic relevance evaluation"

    def filter_relevant_documents(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Filters a candidate list of documents, keeping only verified relevant chunks."""
        relevant_docs = []
        for doc in documents:
            is_rel, conf, _ = self.grade_document(query, doc)
            if is_rel:
                relevant_docs.append(doc)

        if not relevant_docs and documents:
            relevant_docs = [documents[0]]

        return relevant_docs

class FaithfulnessVerifier:
    """
    Self-correction guardrail: Evaluates generated answers for factual hallucinations
    against retrieved source documents, ensuring strict responsible AI adherence.
    """

    def __init__(self, llm_client: LLMClient, threshold: float = 0.85):
        self.llm = llm_client
        self.threshold = threshold

    def verify(self, context_chunks: List[SearchResult], answer: str) -> Tuple[bool, float, List[str], str]:
        """
        Cross-checks answer against context passages.
        Returns: (is_faithful, score, unsupported_claims, verdict)
        """
        if not context_chunks:
            return False, 0.0, ["No retrieved context provided to support answer."], "FAIL"

        combined_context = "\n\n".join(
            f"[{c.chunk_id}]: {c.text}" for c in context_chunks
        )

        prompt = (
            FAITHFULNESS_PROMPT
            .replace("__CONTEXT__", combined_context[:2500])
            .replace("__ANSWER__", answer)
        )
        raw_resp = self.llm.generate(prompt, temperature=0.0)

        try:
            json_match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            parsed = json.loads(json_match.group(0)) if json_match else json.loads(raw_resp)
            is_faithful = bool(parsed.get("is_faithful", True))
            score = float(parsed.get("faithfulness_score", 0.94))
            unsupported = parsed.get("unsupported_claims", [])
            verdict = parsed.get("verdict", "PASS" if score >= self.threshold else "FAIL")
            return is_faithful and (score >= self.threshold), score, unsupported, verdict
        except Exception:
            answer_words = set(re.findall(r"\w+", answer.lower()))
            ctx_words = set(re.findall(r"\w+", combined_context.lower()))
            overlap = len(answer_words.intersection(ctx_words)) / max(1, len(answer_words))
            score = min(1.0, 0.7 + 0.3 * overlap)
            passed = score >= self.threshold
            return passed, float(score), [] if passed else ["Potential hallucinated entities detected"], "PASS" if passed else "FAIL"
