"""Writer agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a clear, concise technical writer. Given research notes and analysis notes, "
    "synthesize a final answer for the given audience. Cite sources inline using [n] "
    "notation matching the numbered source list you are given. Do not invent facts that "
    "are not supported by the notes or sources."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.analysis_notes:
                raise ValueError("writer requires analysis_notes to be populated first")

            sources_block = "\n".join(
                f"[{i}] {doc.title} ({doc.url or 'no url'})"
                for i, doc in enumerate(state.sources, start=1)
            )
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis notes:\n{state.analysis_notes}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Write the final answer now, with inline [n] citations."
            )
            response = self._llm_client.complete(_SYSTEM_PROMPT, user_prompt)

            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writer.complete", {})
        except Exception as exc:  # noqa: BLE001 - convert worker failures into state errors
            message = f"writer failed: {exc}"
            logger.exception(message)
            state.errors.append(message)
            state.add_trace_event("writer.error", {"error": str(exc)})

        return state
