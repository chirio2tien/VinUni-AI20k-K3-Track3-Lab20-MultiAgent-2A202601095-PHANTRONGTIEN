"""Supervisor / router."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (state-machine, no LLM call needed — cheap and deterministic):

    1. No sources yet                    -> researcher
    2. Sources exist, no analysis yet     -> analyst
    3. Analysis exists, no final answer   -> writer
    4. Final answer exists, not reviewed  -> critic
    5. Otherwise                          -> done

    Guardrails:
    - `max_iterations` (from Settings) hard-stops the loop and forces `done` even if the
      state machine above would ask for another step, so a misbehaving/looping agent
      never runs unbounded.
    - If a worker recorded an error for the current step, the supervisor does not retry
      indefinitely: it allows one retry of the same route, then routes past it (fallback)
      to keep the pipeline moving instead of stalling forever.
    """

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        if state.iteration >= self._settings.max_iterations:
            state.add_trace_event(
                "supervisor.stop",
                {"reason": "max_iterations_reached", "iteration": state.iteration},
            )
            state.record_route(ROUTE_DONE)
            return state

        next_route = self._decide_route(state)

        if self._should_fallback(state, next_route):
            next_route = self._fallback_route(next_route)
            state.add_trace_event(
                "supervisor.fallback",
                {"reason": "repeated_failure", "route": next_route},
            )

        state.add_trace_event("supervisor.route", {"next": next_route})
        state.record_route(next_route)
        return state

    def _decide_route(self, state: ResearchState) -> str:
        if not state.sources or not state.research_notes:
            return ROUTE_RESEARCHER
        if not state.analysis_notes:
            return ROUTE_ANALYST
        if not state.final_answer:
            return ROUTE_WRITER
        if not self._has_critic_run(state):
            return ROUTE_CRITIC
        return ROUTE_DONE

    def _has_critic_run(self, state: ResearchState) -> bool:
        return any(result.agent == "critic" for result in state.agent_results)

    def _should_fallback(self, state: ResearchState, next_route: str) -> bool:
        """Detect the same route failing repeatedly and stop retrying it forever."""

        recent_routes = state.route_history[-2:]
        return (
            len(recent_routes) == 2
            and recent_routes[0] == recent_routes[1] == next_route
            and bool(state.errors)
        )

    def _fallback_route(self, failed_route: str) -> str:
        fallback_order = {
            ROUTE_RESEARCHER: ROUTE_ANALYST,
            ROUTE_ANALYST: ROUTE_WRITER,
            ROUTE_WRITER: ROUTE_DONE,
            ROUTE_CRITIC: ROUTE_DONE,
        }
        return fallback_order.get(failed_route, ROUTE_DONE)
