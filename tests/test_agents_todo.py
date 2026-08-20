"""Unit tests for SupervisorAgent's routing policy."""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def _supervisor() -> SupervisorAgent:
    return SupervisorAgent(settings=Settings(max_iterations=6))


def test_routes_to_researcher_when_no_sources() -> None:
    state = _state()
    result = _supervisor().run(state)
    assert result.route_history[-1] == ROUTE_RESEARCHER


def test_routes_to_analyst_when_sources_but_no_analysis() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    result = _supervisor().run(state)
    assert result.route_history[-1] == ROUTE_ANALYST


def test_routes_to_writer_when_analysis_ready() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    result = _supervisor().run(state)
    assert result.route_history[-1] == ROUTE_WRITER


def test_routes_to_critic_after_final_answer() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    result = _supervisor().run(state)
    assert result.route_history[-1] == ROUTE_CRITIC


def test_routes_to_done_after_critic_ran() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state.agent_results.append(AgentResult(agent=AgentName.CRITIC, content="ok"))
    result = _supervisor().run(state)
    assert result.route_history[-1] == ROUTE_DONE


def test_stops_at_max_iterations() -> None:
    state = _state()
    state.iteration = 6
    result = SupervisorAgent(settings=Settings(max_iterations=6)).run(state)
    assert result.route_history[-1] == ROUTE_DONE
