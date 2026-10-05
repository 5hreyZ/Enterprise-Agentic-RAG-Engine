import json
import re
from typing import Dict, Any, List
from enterprise_rag.inference.llm_client import LLMClient
from enterprise_rag.inference.prompts import ROUTER_PROMPT
from enterprise_rag.agents.state import AgentState

class QueryRouter:
    """
    Analyzes enterprise queries to determine if single-hop direct retrieval
    or multi-hop compositional reasoning across disparate topics is required.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def route_and_decompose(self, state: AgentState) -> AgentState:
        """
        Classifies query intent and breaks down complex questions into sub-queries.
        Updates state in-place and returns it.
        """
        prompt = ROUTER_PROMPT.replace("__QUERY__", state.user_query)
        raw_response = self.llm.generate(prompt, temperature=0.0)

        try:
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(raw_response)

            state.is_multi_hop = bool(parsed.get("is_multi_hop", False))
            state.sub_queries = parsed.get("sub_queries", [])
            state.domain_tags = parsed.get("domain_tags", [])
            
            # If marked multi-hop but no sub-queries returned, create atomic steps
            if state.is_multi_hop and not state.sub_queries:
                state.sub_queries = [
                    f"Core parameters and baseline rules: {state.user_query}",
                    f"Specific conditions and comparative metrics: {state.user_query}",
                ]
            elif not state.is_multi_hop:
                state.sub_queries = [state.user_query]

            state.add_trace("query_routing", {
                "is_multi_hop": state.is_multi_hop,
                "sub_queries": state.sub_queries,
                "domain_tags": state.domain_tags,
                "reasoning": parsed.get("reasoning", "Routing complete"),
            })

        except Exception as e:
            # Safe heuristic fallback
            is_complex = any(k in state.user_query.lower() for k in [" and ", "compare", "vs", "versus", "impact of", "relationship between"])
            state.is_multi_hop = is_complex
            if is_complex:
                parts = [p.strip() for p in re.split(r"\band\b|\bvs\b|\bversus\b|\bcompare\b", state.user_query, flags=re.IGNORECASE) if len(p.strip()) > 3]
                state.sub_queries = parts if len(parts) >= 2 else [state.user_query]
            else:
                state.sub_queries = [state.user_query]

            state.add_trace("query_routing_fallback", {
                "is_multi_hop": state.is_multi_hop,
                "sub_queries": state.sub_queries,
                "error": str(e),
            })

        return state
