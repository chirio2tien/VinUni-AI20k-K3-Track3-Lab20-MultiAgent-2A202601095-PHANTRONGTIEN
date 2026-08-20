"""Critic agent: fact-checking and citation-coverage review."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a strict fact-checking critic. Given a final answer and the source material it "
    "was based on, identify any claims that are unsupported, contradicted, or missing a "
    "citation. Be brief: a short bullet list of findings, or 'No issues found.'"
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CriticAgent(BaseAgent):
    """Fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.final_answer:
                raise ValueError("critic requires final_answer to be populated first")

            citation_coverage = self._citation_coverage(state)

            sources_block = "\n".join(
                f"[{i}] {doc.title} — {doc.snippet}" for i, doc in enumerate(state.sources, start=1)
            )
            user_prompt = (
                f"Final answer:\n{state.final_answer}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Review the final answer for unsupported claims or missing citations."
            )
            response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "citation_coverage": citation_coverage,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "critic.complete", {"citation_coverage": citation_coverage}
            )
        except Exception as exc:  # noqa: BLE001 - convert worker failures into state errors
            message = f"critic failed: {exc}"
            logger.exception(message)
            state.errors.append(message)
            state.add_trace_event("critic.error", {"error": str(exc)})

        return state

    def _citation_coverage(self, state: ResearchState) -> float:
        """Fraction of sentences in the final answer that carry a [n] citation."""

        if not state.final_answer:
            return 0.0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", state.final_answer) if s.strip()]
        if not sentences:
            return 0.0
        cited = sum(1 for s in sentences if _CITATION_PATTERN.search(s))
        return cited / len(sentences)
