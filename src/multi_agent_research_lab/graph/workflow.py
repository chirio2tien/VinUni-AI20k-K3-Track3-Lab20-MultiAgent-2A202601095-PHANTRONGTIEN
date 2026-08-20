"""LangGraph workflow for the multi-agent research pipeline."""

import logging
import uuid
from collections.abc import Iterator
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover - exercised when the `llm` extra isn't installed
    _HAS_LANGGRAPH = False


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.

    Nodes: supervisor, researcher, analyst, writer, critic.
    The supervisor node writes the next route into `state.route_history[-1]`; a conditional
    edge reads that value to decide which worker node runs next, or routes to END.
    """

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self._graph: Any | None = None
        self._checkpointer: object | None = None
        self._agents = {
            "supervisor": self.supervisor,
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

    def _run_node(self, node_name: str, state: ResearchState) -> ResearchState:
        with trace_span(f"agent.{node_name}", {"iteration": state.iteration}):
            return self._agents[node_name].run(state)

    def build(self) -> object:
        """Create a LangGraph graph with conditional routing driven by the supervisor."""

        if not _HAS_LANGGRAPH:
            logger.info("langgraph not installed; MultiAgentWorkflow will use the fallback loop")
            self._graph = None
            return None

        builder: Any = StateGraph(ResearchState)
        for node_name in self._agents:
            builder.add_node(node_name, lambda s, n=node_name: self._run_node(n, s))

        builder.set_entry_point("supervisor")

        def route_after_supervisor(state: ResearchState) -> str:
            return state.route_history[-1] if state.route_history else ROUTE_DONE

        builder.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                ROUTE_RESEARCHER: "researcher",
                ROUTE_ANALYST: "analyst",
                ROUTE_WRITER: "writer",
                ROUTE_CRITIC: "critic",
                ROUTE_DONE: END,
            },
        )
        for worker in ("researcher", "analyst", "writer", "critic"):
            builder.add_edge(worker, "supervisor")

        self._checkpointer = MemorySaver()
        self._graph = builder.compile(checkpointer=self._checkpointer)
        return self._graph

    def run(self, state: ResearchState, thread_id: str | None = None) -> ResearchState:
        """Execute the graph (or fallback loop) and return the final state."""

        final_state = state
        for emitted in self.run_streaming(state, thread_id=thread_id):
            final_state = emitted
        return final_state

    def run_streaming(
        self, state: ResearchState, thread_id: str | None = None
    ) -> Iterator[ResearchState]:
        """Execute the workflow, yielding the state after every node completes.

        Lets a caller (e.g. the CLI) render progress in real time instead of blocking
        until the whole pipeline finishes. Uses LangGraph's `.stream()` API when available,
        falling back to a plain-Python loop that yields the same way otherwise.

        `thread_id` identifies the run in the checkpointer: passing the same id for a
        retry resumes from the last completed node instead of starting over (LangGraph
        path only; the fallback loop always restarts from the given `state`).
        """

        if self._graph is None:
            self.build()

        if self._graph is not None:
            yield from self._stream_langgraph(state, thread_id or str(uuid.uuid4()))
        else:
            yield from self._stream_fallback_loop(state)

    def _stream_langgraph(self, state: ResearchState, thread_id: str) -> Iterator[ResearchState]:
        assert self._graph is not None
        config = {"recursion_limit": 100, "configurable": {"thread_id": thread_id}}
        for chunk in self._graph.stream(state, config=config):
            node_output = next(iter(chunk.values()))
            if isinstance(node_output, ResearchState):
                state = node_output
            else:
                state = ResearchState(**node_output)
            yield state

    def _stream_fallback_loop(self, state: ResearchState) -> Iterator[ResearchState]:
        """Plain-Python state machine used when `langgraph` isn't installed."""

        worker_routes = {ROUTE_RESEARCHER, ROUTE_ANALYST, ROUTE_WRITER, ROUTE_CRITIC}
        while True:
            state = self._run_node("supervisor", state)
            yield state
            route = state.route_history[-1] if state.route_history else ROUTE_DONE
            if route == ROUTE_DONE or route not in worker_routes:
                return
            state = self._run_node(route, state)
            yield state
