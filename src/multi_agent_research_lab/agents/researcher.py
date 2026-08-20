"""Researcher agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a meticulous research agent. Given a user query and a list of retrieved "
    "sources, write concise research notes (bullet points) that capture the key facts. "
    "Reference sources by their index, e.g. [1], [2], so claims can be traced back."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._search_client = search_client or SearchClient()
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            sources = self._search_client.search(
                state.request.query, max_results=state.request.max_sources
            )
            state.sources = sources

            sources_block = "\n".join(
                f"[{i}] {doc.title} — {doc.snippet}" for i, doc in enumerate(sources, start=1)
            )
            user_prompt = (
                f"Query: {state.request.query}\n\nSources:\n{sources_block}\n\n"
                "Write research notes summarizing the most relevant facts from these sources."
            )
            response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)

            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "source_count": len(sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "researcher.complete", {"source_count": len(sources)}
            )
        except Exception as exc:  # noqa: BLE001 - convert worker failures into state errors
            message = f"researcher failed: {exc}"
            logger.exception(message)
            state.errors.append(message)
            state.add_trace_event("researcher.error", {"error": str(exc)})

        return state
