"""Analyst agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, AnalysisResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an analytical agent. Given research notes, extract the key claims, compare "
    "differing viewpoints if any, and flag claims with weak or missing evidence. For each "
    "claim, list which source indices (matching the [n] notation in the notes) support it, "
    "and a confidence score from 0 to 1."
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
            analysis, response = self._llm_client.complete_structured(
                _SYSTEM_PROMPT, user_prompt, AnalysisResult
            )

            state.analysis = analysis
            state.analysis_notes = _render_analysis(analysis)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=state.analysis_notes,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "claim_count": len(analysis.key_claims),
                    },
                )
            )
            state.add_trace_event("analyst.complete", {"claim_count": len(analysis.key_claims)})
        except Exception as exc:  # noqa: BLE001 - convert worker failures into state errors
            message = f"analyst failed: {exc}"
            logger.exception(message)
            state.errors.append(message)
            state.add_trace_event("analyst.error", {"error": str(exc)})

        return state


def _render_analysis(analysis: AnalysisResult) -> str:
    lines = ["Key claims:"]
    for claim in analysis.key_claims:
        sources = ", ".join(f"[{i}]" for i in claim.supporting_source_indices) or "no source"
        lines.append(f"- {claim.text} ({sources}, confidence={claim.confidence:.2f})")

    if analysis.conflicting_viewpoints:
        lines.append("\nConflicting viewpoints:")
        lines.extend(f"- {v}" for v in analysis.conflicting_viewpoints)

    if analysis.weak_evidence_flags:
        lines.append("\nWeak evidence flags:")
        lines.extend(f"- {v}" for v in analysis.weak_evidence_flags)

    if analysis.summary:
        lines.append(f"\nSummary: {analysis.summary}")

    return "\n".join(lines)
