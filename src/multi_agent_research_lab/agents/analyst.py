"""Analyst agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an analytical agent. Given research notes, extract the key claims, compare "
    "differing viewpoints if any, and flag claims with weak or missing evidence. Produce "
    "structured analysis notes (bullet points) that a writer can turn into a final answer."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.research_notes:
                raise ValueError("analyst requires research_notes to be populated first")

            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Extract key claims, compare viewpoints, and flag weak evidence."
            )
            response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)

            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst.complete", {})
        except Exception as exc:  # noqa: BLE001 - convert worker failures into state errors
            message = f"analyst failed: {exc}"
            logger.exception(message)
            state.errors.append(message)
            state.add_trace_event("analyst.error", {"error": str(exc)})

        return state
